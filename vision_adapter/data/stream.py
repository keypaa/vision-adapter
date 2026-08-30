"""vision_adapter/data/stream.py — HF streaming for embeddings.

Canonical streaming stack extracted from grok_probe_qwen.py so both
`vision_adapter.train` and `grok_probe_qwen.py` share one implementation.
No Modal dependency; any CUDA host streams the 103 HF parquet shards via
HTTPS Range requests (≈400MiB per row group, pipelined).

Contracts pinned by test_local_pack.py:
  vis_bytes = torch bf16 -> bytes, reconstructed via
  np.frombuffer(vis_bytes, uint8).view(bfloat16).reshape(-1, 4096).float()
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from vision_adapter.backends.auth import get_hf_token as _get_hf_token

EMB_REPO = "keypa/vision-adapter-embeddings"
MANIFEST_REPO = "keypa/vision-adapter-manifests"
MANIFEST_FILE = "train_manifest.jsonl"
KEY_INDEX_CACHE = "key_index_cache.json"

# Vision dim is fixed by MoonViT 2x2 merge flatten — matches config TrainConfig.vision_dim
VISION_DIM = 4096


def _auth_headers() -> dict[str, str]:
    tok = _get_hf_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _remote_size(url: str) -> int:
    import urllib.request

    req = urllib.request.Request(url, method="HEAD", headers=_auth_headers())
    return int(urllib.request.urlopen(req).headers["Content-Length"])


def _fetch_range(url: str, start: int, end: int, retries: int = 3) -> bytes:
    import urllib.request

    headers = dict(_auth_headers())
    headers["Range"] = f"bytes={start}-{end}"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req).read()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(0.5 * (2**attempt))
    raise last_err  # type: ignore[misc]


class RemoteShard(io.RawIOBase):
    """Remote parquet shard seekable via HTTPS Range.

    Only the footer (64 KiB) and the currently pre-fetched row-group span
    are served; pyarrow's ParquetFile only seeks to those two regions.
    """

    FOOTER_BYTES = 64 * 2**10
    FETCH_CHUNK = 16 * 2**20
    N_STREAMS = 4

    def __init__(self, url: str, size: int, disk_cache: str | None = None):
        super().__init__()
        self.url, self.size = url, size
        self.pos = 0
        self.disk_cache = disk_cache
        self.footer = _fetch_range(url, size - self.FOOTER_BYTES, size - 1)
        self._span: tuple[int, bytes] | None = None

    def _cache_file(self, lo: int, hi: int) -> str:
        h = hashlib.sha1(f"{self.url}:{lo}:{hi}".encode()).hexdigest()[:20]
        return os.path.join(self.disk_cache, f"rg_{h}.bin")  # type: ignore[arg-type]

    def load_span(self, lo: int, hi: int) -> None:
        if self._span is not None and self._span[0] == lo:
            return
        if self.disk_cache:
            cf = self._cache_file(lo, hi)
            if os.path.exists(cf):
                with open(cf, "rb") as f:
                    blob = f.read()
                if len(blob) == hi - lo:
                    self._span = (lo, blob)
                    return
                os.remove(cf)
        bounds = [(st, min(st + self.FETCH_CHUNK, hi)) for st in range(lo, hi, self.FETCH_CHUNK)]
        with ThreadPoolExecutor(self.N_STREAMS) as ex:
            futs = {ex.submit(_fetch_range, self.url, st, en - 1): (st, en) for st, en in bounds}
            got = {st: fut.result() for fut, (st, en) in futs.items()}
        blob = b"".join(got[st] for st, _ in bounds)
        if self.disk_cache and len(blob) == hi - lo:
            tmp = self._cache_file(lo, hi) + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(blob)
                os.replace(tmp, self._cache_file(lo, hi))
            except Exception:
                pass
        self._span = (lo, blob)

    def seek(self, off, whence=0):  # type: ignore[override]
        self.pos = off if whence == 0 else (self.size + off if whence == 2 else self.pos)
        return self.pos

    def tell(self):  # type: ignore[override]
        return self.pos

    def readable(self):  # type: ignore[override]
        return True

    def seekable(self):  # type: ignore[override]
        return True

    def read(self, n=-1):  # type: ignore[override]
        if n == -1:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        if self.pos >= self.size - len(self.footer):
            off = self.pos - (self.size - len(self.footer))
            chunk = self.footer[off : off + (end - self.pos)]
        elif self._span is not None:
            lo, blob = self._span
            if lo <= self.pos < lo + len(blob):
                chunk = blob[self.pos - lo : self.pos - lo + (end - self.pos)]
            else:
                raise OSError(f"read at {self.pos} outside span [{lo}, {lo + len(blob)})")
        else:
            raise OSError("no row group pre-fetched")
        self.pos += len(chunk)
        return chunk


def rg_span(md, rg_idx: int, columns: tuple[str, ...] | None = None) -> tuple[int, int]:
    rg = md.row_group(rg_idx)
    names = [md.row_group(rg_idx).column(c).path_in_schema for c in range(rg.num_columns)]
    starts, ends = [], []
    for c in range(rg.num_columns):
        if columns is not None and names[c] not in columns:
            continue
        col = rg.column(c)
        st = getattr(col, "dictionary_page_offset", None)
        if st is None:
            st = col.data_page_offset
        starts.append(st)
        ends.append(st + col.total_compressed_size)
    return min(starts), max(ends)


def shard_row_group_size(md) -> int:
    return max(md.row_group(g).num_rows for g in range(md.num_row_groups))


def _prefetch_key_spans(url: str, md) -> dict[int, bytes]:
    spans = [(rgi, *rg_span(md, rgi, columns=("key",))) for rgi in range(md.num_row_groups)]
    with ThreadPoolExecutor(max_workers=min(12, len(spans))) as ex:
        futs = {ex.submit(_fetch_range, url, lo, hi - 1): rgi for rgi, lo, hi in spans}
        return {futs[fut]: fut.result() for fut in as_completed(futs)}


def _cache_key_index_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, KEY_INDEX_CACHE)


def save_key_index(index: dict[str, tuple[str, int]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 2, "keys": {k: {"shard": s, "row": r} for k, (s, r) in index.items()}}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def load_key_index(path: str) -> tuple[dict[str, tuple[str, int]], bool]:
    if not os.path.exists(path):
        return {}, False
    try:
        with open(path) as f:
            payload = json.load(f)
        if payload.get("version") != 2 or not isinstance(payload.get("keys"), dict):
            return {}, False
        return {k: (v["shard"], v["row"]) for k, v in payload["keys"].items()}, True
    except Exception:
        return {}, False


def list_shards(token: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token or _get_hf_token())
    files = api.list_repo_files(EMB_REPO, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def fetch_manifest(cache_dir: str | None = None, token: str | None = None) -> list[dict]:
    """Download + parse train_manifest.jsonl ({emb, user, assistant, g}), skipping header."""
    from huggingface_hub import hf_hub_download

    tok = token or _get_hf_token()
    kw = dict(local_dir=cache_dir) if cache_dir else {}
    if tok:
        kw["token"] = tok  # type: ignore[assignment]
    local = hf_hub_download(MANIFEST_REPO, MANIFEST_FILE, repo_type="dataset", **kw)
    rows: list[dict] = []
    with open(local) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if obj.get("type") == "manifest_header":
                    continue
                rows.append(obj)
    return rows


def build_key_index(
    stream_order: list[str], cache_dir: str | None = None, rebuild: bool = False
) -> dict[str, tuple[str, int]]:
    """emb key -> (shard_file, global_row_idx), footer+key-chunk only, cached."""
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key_index_path(cache_dir or ".")
    if not rebuild:
        index, ok = load_key_index(cache_path)
        if ok and len(index) > 0:
            print(f"[stream] key index loaded from cache ({len(index)} embeddings)", flush=True)
            return index
    index: dict[str, tuple[str, int]] = {}
    t0 = time.time()
    import pyarrow.parquet as pq

    for i, sf in enumerate(stream_order):
        url = f"https://huggingface.co/datasets/{EMB_REPO}/resolve/main/{sf}"
        rs = RemoteShard(url, _remote_size(url))
        pf = pq.ParquetFile(rs)
        md = pf.metadata
        spans = _prefetch_key_spans(url, md)
        row_cursor = 0
        for rgi in range(md.num_row_groups):
            lo, _ = rg_span(md, rgi, columns=("key",))
            rs._span = (lo, spans[rgi])
            tbl = pf.read_row_group(rgi, columns=["key"])
            keys_here = tbl.column("key").to_pylist()
            g0 = md.row_group(rgi).num_rows
            for j, k in enumerate(keys_here):
                if k in index:
                    raise AssertionError(f"duplicate embedding key: {k}")
                index[k] = (sf, row_cursor + j)
            row_cursor += g0
        print(f"[stream]   key index {i+1}/{len(stream_order)} {Path(sf).name} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[stream] key index complete: {len(index)} embeddings in {time.time()-t0:.0f}s", flush=True)
    if cache_dir:
        save_key_index(index, cache_path)
        print(f"[stream] key index cached to {cache_path}", flush=True)
    return index


def build_epoch_plan(
    rows: list[dict],
    index: dict[str, tuple[str, int]],
    sample_size: int,
    seed: int,
    excluded_shards: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Cluster sampling: seed-shuffle shards, take whole shards until budget."""
    rng = random.Random(seed)
    excluded = {f"datasets/{s}" if "/" in s and not s.startswith("data/") else s for s in (excluded_shards or set())}
    by_shard_rows: dict[str, list[tuple[int, dict]]] = {}
    for r in rows:
        loc = index.get(r.get("emb"))
        if loc is None or loc[0] in excluded:
            continue
        by_shard_rows.setdefault(loc[0], []).append((loc[1], r))
    n_available = sum(len(v) for v in by_shard_rows.values())
    dropped = len(rows) - n_available
    if dropped:
        print(f"[stream] NOTE: {dropped} manifest rows skipped (no embedding or excluded)", flush=True)
    shard_files = sorted(by_shard_rows)
    rng.shuffle(shard_files)
    plan: dict[str, list[dict]] = {}
    budget = sample_size
    for sf in shard_files:
        if budget <= 0:
            break
        take = min(budget, len(by_shard_rows[sf]))
        selected = sorted(by_shard_rows[sf], key=lambda t: t[0])[:take]
        plan[sf] = [{**r, "_row": j} for j, r in selected]
        budget -= take
    print(f"[stream] cluster plan: {sum(len(v) for v in plan.values())} rows from {len(plan)} shards", flush=True)
    return plan


class EmbStreamDataset(torch.utils.data.IterableDataset):
    """Shard-by-shard streaming over HTTPS with background prefetch."""

    PREFETCH_DEPTH = 2
    MAX_RG_ROWS = 128

    def __init__(
        self,
        plan: dict[str, list[dict]],
        stream_order: list[str],
        start_pos: int = 0,
        rg_cache_dir: str | None = None,
        vision_dim: int = VISION_DIM,
    ):
        super().__init__()
        self.plan, self.order = plan, stream_order
        self.start_pos = start_pos
        self.rg_cache_dir = rg_cache_dir
        self.vision_dim = vision_dim
        if rg_cache_dir:
            os.makedirs(rg_cache_dir, exist_ok=True)

    def __iter__(self):
        import pyarrow.parquet as pq

        emitted = 0
        for sf in self.order:
            rows_here = self.plan.get(sf)
            if not rows_here:
                continue
            url = f"https://huggingface.co/datasets/{EMB_REPO}/resolve/main/{sf}"
            rs = RemoteShard(url, _remote_size(url), disk_cache=self.rg_cache_dir)
            pf = pq.ParquetFile(rs)
            md = pf.metadata
            biggest = shard_row_group_size(md)
            assert biggest <= self.MAX_RG_ROWS, (
                f"{sf}: row group {biggest} rows (~9 GiB) not streamable — exclude smoke shards"
            )
            g0 = md.row_group(0).num_rows or 64
            want: dict[int, dict[int, dict]] = {}
            for r in rows_here:
                want.setdefault(r["_row"] // g0, {})[r["_row"] % g0] = r
            for rgi in sorted(want):
                lo, hi = rg_span(md, rgi)
                t0 = time.time()
                rs.load_span(lo, hi)
                dt = time.time() - t0
                if dt > 2:
                    print(f"[stream] streamed {Path(sf).name} rg{rgi} ({(hi-lo)/2**20:.0f}MiB in {dt:.0f}s)", flush=True)
                tbl = pf.read_row_group(rgi, columns=["key", "n_vis", "vis_bytes"])
                try:
                    for j in sorted(want[rgi]):
                        row = want[rgi][j]
                        if emitted < self.start_pos:
                            emitted += 1
                            continue
                        vb = tbl.column("vis_bytes")[j].as_py()
                        nv = int(tbl.column("n_vis")[j].as_py())
                        buf = bytearray(vb)
                        del vb
                        vis = (
                            torch.from_numpy(np.frombuffer(buf, dtype=np.uint8))
                            .view(torch.bfloat16)
                            .reshape(-1, self.vision_dim)
                            .float()
                        )
                        del buf
                        assert vis.shape == (nv, self.vision_dim), f"schema mismatch {row['emb']}"
                        yield {"vis": vis, "user": row["user"], "assistant": row["assistant"], "g": row.get("g", "?")}
                        emitted += 1
                finally:
                    del tbl
                    rs._span = None
