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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from vision_adapter.backends.auth import get_hf_token as _get_hf_token

EMB_REPO = "keypa/vision-adapter-embeddings"
MANIFEST_REPO = "keypa/vision-adapter-manifests"
# The backfilled manifest, which carries grid_thw on every row (verified
# 117600/117600 against the stored n_vis). Preferred over the plain one
# when present; see train.resolve_manifest_name.
MANIFEST_FILE = "train_manifest_grids.jsonl"
# Shape of the row stream the loader serves, recorded in every checkpoint.
#   v1 = one row per image position (the manifest's extra questions were dropped)
#   v2 = every plan row is served (the questions of an image are interleaved)
# A resume across versions is refused: the row at a given offset is a different
# row, so the "data continues where it left off" contract would be broken.
STREAM_VERSION = 2
KEY_INDEX_CACHE = "key_index_cache.json"

# Vision dim is fixed by MoonViT 2x2 merge flatten — matches config TrainConfig.vision_dim
VISION_DIM = 4096


def _auth_headers() -> dict[str, str]:
    tok = _get_hf_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _remote_size(url: str) -> int:
    import urllib.error
    import urllib.request

    t0 = time.time()
    last_err = None
    for attempt in range(1 + _RATE_LIMIT_RETRIES):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=_auth_headers())
            size = int(urllib.request.urlopen(req, timeout=30).headers["Content-Length"])
            break
        except urllib.error.HTTPError as e:
            last_err = e
            if getattr(e, "code", None) == 429 and attempt < _RATE_LIMIT_RETRIES:
                time.sleep(_retry_after_s(getattr(e, "headers", None)))
                continue
            raise
    else:
        raise last_err  # type: ignore[misc]
    dt = time.time() - t0
    if dt > 5:
        print(f"[stream] HEAD slow: {Path(url).name} {size/2**30:.1f}GiB in {dt:.0f}s", flush=True)
    return size


_RATE_LIMIT_RETRIES = 8
_RATE_LIMIT_DEFAULT_S = 30.0
_RATE_LIMIT_MAX_S = 300.0


def _retry_after_s(headers, default: float = _RATE_LIMIT_DEFAULT_S) -> float:
    """Parse Retry-After (seconds) with safe default + cap. Never raises."""
    try:
        if headers is not None:
            v = headers.get("Retry-After", None)
            if v is not None:
                return min(max(float(str(v).strip()), 0.0), _RATE_LIMIT_MAX_S)
    except Exception:
        pass
    return default


def _fetch_range(url: str, start: int, end: int, retries: int = 3) -> bytes:  # noqa: C901
    import random
    import urllib.error
    import urllib.request
    from http.client import IncompleteRead

    headers = dict(_auth_headers())
    headers["Range"] = f"bytes={start}-{end}"
    last_err = None
    attempt = 0
    rate_limited = 0
    while True:
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            # 429 storms (parallel Range/index builds) get their own budget with
            # server-honoring backoff — the generic 3-attempt budget can't survive them.
            last_err = e
            if getattr(e, "code", None) == 429:
                if rate_limited >= _RATE_LIMIT_RETRIES:
                    break  # honored the server 8x over minutes — give up loudly
                rate_limited += 1
                time.sleep(_retry_after_s(getattr(e, "headers", None)) + random.uniform(0, 5))
                continue
            if attempt >= retries - 1:
                break
            time.sleep(0.5 * (2**attempt))
        except IncompleteRead as e:
            # HF CDN truncated mid-chunk (common under chunked Range) — retry fresh TCP
            last_err = e
            # Discard partial, retry with backoff; avoid tight loop on 20MiB+ truncations
            if attempt >= retries - 1:
                break
            time.sleep(1.0 * (2**attempt))
        except Exception as e:
            last_err = e
            if attempt >= retries - 1:
                break
            time.sleep(0.5 * (2**attempt))
        attempt += 1
    raise last_err  # type: ignore[misc]


class RemoteShard(io.RawIOBase):
    """Remote parquet shard seekable via HTTPS Range.

    Only the footer (64 KiB) and the currently pre-fetched row-group span
    are served; pyarrow's ParquetFile only seeks to those two regions.
    """

    FOOTER_BYTES = 64 * 2**10
    FETCH_CHUNK = 32 * 2**20
    N_STREAMS = 8

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
        print(f"[stream] load_span {Path(self.url).name} {(hi-lo)/2**20:.0f}MiB @{lo}", flush=True)
        if self.disk_cache:
            cf = self._cache_file(lo, hi)
            if os.path.exists(cf):
                try:
                    sz = os.path.getsize(cf)
                    if sz == hi - lo:
                        with open(cf, "rb") as f:
                            blob = f.read()
                        if len(blob) == hi - lo:
                            self._span = (lo, blob)
                            return
                    os.remove(cf)
                except Exception:
                    pass
        # One session per span load — avoids urllib's per-chunk TCP setup.
        # Keep N_STREAMS chunked Range (HF CDN rewards multi-connection)
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
                _enforce_rg_cache_limit(self.disk_cache)
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
    spans = [(rgi, *rg_span(md, rgi, columns=("key", "n_vis"))) for rgi in range(md.num_row_groups)]
    with ThreadPoolExecutor(max_workers=min(12, len(spans))) as ex:
        futs = {ex.submit(_fetch_range, url, lo, hi - 1): rgi for rgi, lo, hi in spans}
        return {futs[fut]: fut.result() for fut in as_completed(futs)}


def _cache_key_index_path(cache_dir: str, stream_order: list[str] | None = None) -> str:
    """Cache file for a key index, namespaced by shard set when known.

    A shared filename across different shard sets silently serves the wrong
    index (eval over 2 shards once loaded a 13-shard training index).
    """
    if not stream_order:
        return os.path.join(cache_dir, KEY_INDEX_CACHE)
    import hashlib

    digest = hashlib.sha1("\n".join(sorted(stream_order)).encode()).hexdigest()[:12]
    stem = KEY_INDEX_CACHE.removesuffix(".json")
    return os.path.join(cache_dir, f"{stem}.{digest}.json")


def save_key_index(index: dict[str, tuple[str, int] | tuple[str, int, int]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 3, "keys": {}}
    for k, v in index.items():
        if len(v) == 2:
            payload["keys"][k] = {"shard": v[0], "row": v[1]}
        else:
            payload["keys"][k] = {"shard": v[0], "row": v[1], "n_vis": v[2]}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def load_key_index(path: str) -> tuple[dict[str, tuple[str, int] | tuple[str, int, int]], bool]:
    if not os.path.exists(path):
        return {}, False
    try:
        with open(path) as f:
            payload = json.load(f)
        ver = payload.get("version")
        if ver not in (2, 3) or not isinstance(payload.get("keys"), dict):
            return {}, False
        out: dict[str, tuple[str, int] | tuple[str, int, int]] = {}
        for k, v in payload["keys"].items():
            if ver == 3 and "n_vis" in v:
                out[k] = (v["shard"], v["row"], v["n_vis"])
            else:
                out[k] = (v["shard"], v["row"])
        return out, True  # type: ignore[return-value]
    except Exception:
        return {}, False


def _in_modal() -> bool:
    return bool(os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_ENVIRONMENT"))


def _get_hf_shard_path(shard: str, cache_dir: str | None = None) -> str | None:
    """If a whole shard is already on this disk, return its path.

    The download keeps the repo-relative path, so the file lands at
    <cache_dir>/data/emb_XXXX.parquet -- the same place _shard_cache_path
    looks. (The old lookup only looked at <cache_dir>/emb_*.parquet, so
    every shard looked "missing" and cost an extra round trip.)
    """
    if cache_dir:
        p = _shard_cache_path(cache_dir, shard)
        if p:
            return p
    for base in (os.environ.get("HF_HUB_CACHE"), os.path.expanduser("~/.cache/huggingface/hub")):
        if not base:
            continue
        import glob

        hits = glob.glob(os.path.join(base, "**", os.path.basename(shard)), recursive=True)
        if hits:
            return hits[0]
    return None


def _download_shard_hf_transfer(shard: str, cache_dir: str | None = None, token: str | None = None) -> str | None:
    """Whole-shard download via hf_transfer (Rust, ~1GiB/s aggregate).

    Was Modal-only (`if not _in_modal(): return None`), which meant the target
    platform — a rented GPU instance — always paid the urllib Range path,
    measured at 1.5-28 MiB/s against hf_transfer's documented ~1 GiB/s.
    Two independent audits flagged the same gate.

    Returns local path on success, None on miss/failure (caller falls back to
    Range). Identical bytes and identical row order as the Range path, so
    checkpoints and resume are unaffected — only the transport changes.
    Enable the Rust transfer with HF_HUB_ENABLE_HF_TRANSFER=1; without
    huggingface_hub it degrades to the Range path as before.
    """
    if cache_dir:
        # already downloaded? hand it back without touching the network
        cached = _shard_cache_path(cache_dir, shard)
        if cached:
            return cached
        # Refuse a download that cannot fit: the shards are 0.4-60 GB and the
        # disk budget is half the free space. A shard that does not fit is
        # served by the Range path instead of filling the disk.
        try:
            size = _remote_size(f"https://huggingface.co/datasets/{EMB_REPO}/resolve/main/{shard}")
            budget = _shard_cache_budget_bytes(cache_dir)
            cached_bytes = sum(os.path.getsize(p) for p in _cached_shard_files(cache_dir))
            if size > budget - cached_bytes:
                print(f"[stream] shard {shard} is {size/2**30:.1f}GiB and the cache budget "
                      f"is {budget/2**30:.1f}GiB (free {cached_bytes/2**30:.1f}GiB used) — "
                      f"skipping the whole-shard download, using Range", flush=True)
                return None
        except Exception:
            pass
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]

        tok = token or _get_hf_token()
        kw: dict[str, object] = {"repo_id": EMB_REPO, "repo_type": "dataset", "filename": shard}
        if tok:
            kw["token"] = tok  # type: ignore[assignment]
        if cache_dir:
            kw["local_dir"] = cache_dir  # type: ignore[assignment]
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(**kw)  # type: ignore[arg-type]
        return p if os.path.exists(p) else None
    except Exception:
        return None


class _DaemonTask:
    """A single background task on a DAEMON thread.

    ThreadPoolExecutor registers an atexit hook that JOINS its workers, so a
    60 GB shard download still in flight when the trainer stops would hang
    the process for minutes. A daemon thread can be abandoned.
    """

    def __init__(self, fn, *args, **kwargs):
        self.result = None
        self.error: Exception | None = None
        self._done = threading.Event()

        def _run():
            try:
                self.result = fn(*args, **kwargs)
            except Exception as e:      # noqa: BLE001 - reported, not raised
                self.error = e
            finally:
                self._done.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def value(self):
        if self.error is not None:
            raise self.error
        return self.result


def drain_row_group_prefetch(fut):
    """Wait for a pending row-group prefetch and return its fetched span.

    Without this the main thread calls `load_span` on a span the prefetch
    thread is still fetching: the cache file is not there yet, so the same
    bytes are downloaded twice, concurrently. That is the worst place to
    double traffic — it happens precisely on the slow row groups the prefetch
    was introduced to hide.

    Returns None when nothing is in flight (the common case) so the caller
    falls through to its normal load.
    """
    if fut is None:
        return None
    try:
        return fut.result()
    except Exception:
        return None


def _cached_shard_files(cache_dir: str) -> list[str]:
    """Whole-shard files in the cache, wherever they actually live.

    `hf_hub_download(local_dir=...)` keeps the repo-relative path, so a shard
    downloaded as `data/emb_0094.parquet` lands at
    `<cache_dir>/data/emb_0094.parquet`, NOT `<cache_dir>/emb_0094.parquet`.
    The old glob (`<cache_dir>/emb_*.parquet`) therefore matched nothing and
    the whole-shard cache was never evicted: it grew without bound on a big
    box and filled the disk on a small one.
    """
    import glob

    pats = [
        os.path.join(cache_dir, "**", "emb_*.parquet"),
        os.path.join(cache_dir, "emb_*.parquet"),
    ]
    out: list[str] = []
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            if os.path.isfile(p) and p not in out:
                out.append(p)
    return out


SHARD_CACHE_FREE_FRACTION = 0.5   # the whole-shard cache may use half the disk


def _shard_cache_budget_bytes(cache_dir: str) -> int:
    """How many bytes the whole-shard cache may hold on this host.

    A byte budget, not a shard count: the shards are 0.4-60 GB, so any fixed
    count is either a disk-fill hazard or a thrash. Half the free disk, so
    there is room for the row-group cache and the run's own output.
    """
    try:
        import shutil

        free = shutil.disk_usage(cache_dir).free
        return max(1 * 2**30, int(free * SHARD_CACHE_FREE_FRACTION))
    except Exception:
        return 1 * 2**30


def _enforce_lru_cache(cache_dir: str | None, max_shards: int | None = None) -> None:
    """Evict whole-shard cache files (oldest first) until the byte budget holds.

    The whole-shard path is open on every host now, so the cache is a real
    disk consumer: 0.4-60 GB per shard. The budget is derived from the free
    disk; ``max_shards`` (when given) is an additional ceiling.
    """
    if not cache_dir or not os.path.isdir(cache_dir):
        return
    try:
        shards = _cached_shard_files(cache_dir)
        if not shards:
            return
        budget = _shard_cache_budget_bytes(cache_dir)
        if max_shards is not None:
            try:
                biggest = max((os.path.getsize(p) for p in shards), default=0)
                budget = min(budget, int(max_shards) * biggest)
            except (TypeError, ValueError):
                pass
        shards.sort(key=lambda p: os.path.getmtime(p))
        total = sum(os.path.getsize(p) for p in shards)
        while total > budget and len(shards) > 1:
            old = shards.pop(0)
            try:
                sz = os.path.getsize(old)
                os.remove(old)
                total -= sz
                print(f"[stream] LRU evicted {os.path.basename(old)} "
                      f"({sz/2**20:.0f}MiB, cache budget {budget/2**30:.0f}GiB)", flush=True)
            except Exception:
                pass
    except Exception:
        pass


def _shard_lru_limit(cache_dir: str, max_shards: int | None = None) -> int:
    """How many whole shards fit in the disk budget for ``cache_dir``.

    80% of free space, so the download that triggers the eviction still lands.
    Falls back to 4 when the disk cannot be measured — the previous
    behaviour, which is the safe side for a small disk.
    """
    if max_shards is not None:
        try:
            return max(1, int(max_shards))
        except (TypeError, ValueError):
            pass
    try:
        import shutil

        free = shutil.disk_usage(cache_dir).free
        biggest = _biggest_cached_shard(cache_dir)
        if biggest <= 0:
            return 4
        return max(1, int((free * 0.8) // biggest))
    except Exception:
        return 4


def _biggest_cached_shard(cache_dir: str) -> int:
    """Largest cached shard in bytes, or 0 — the unit the disk budget divides by."""
    try:
        sizes = [os.path.getsize(p) for p in _cached_shard_files(cache_dir)]
        return max(sizes) if sizes else 0
    except Exception:
        return 0


def _shard_cache_path(cache_dir: str | None, shard: str) -> str | None:
    """Local path of an already-downloaded whole shard, or None."""
    if not cache_dir:
        return None
    for cand in (
        os.path.join(cache_dir, shard),
        os.path.join(cache_dir, os.path.basename(shard)),
    ):
        if os.path.exists(cand):
            return cand
    return None


def _coalesce_ranges(ranges: list[tuple[int, int]], gap: int = 2 * 2**20) -> list[tuple[int, int]]:
    """Coalesce sorted [lo,hi] ranges when gap <=2MiB (Phase 3 micro-Range).

    Merges nearby per-row slices into one Range request to avoid 8× tiny fetches.
    Used for Colab 12GiB path when manifest has vis_off/len (future)."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    out = [ranges[0]]
    for lo, hi in ranges[1:]:
        plo, phi = out[-1]
        if lo - phi <= gap:
            out[-1] = (plo, max(phi, hi))
        else:
            out.append((lo, hi))
    return out


RG_CACHE_MIN_BYTES = 12 * 2**30      # floor: the old Colab-sized budget
RG_CACHE_MAX_BYTES = 64 * 2**30      # ceiling: beyond this it is not a cache
RG_CACHE_FREE_FRACTION = 0.10         # ... and it never exceeds this of the disk


def _rg_cache_budget(cache_dir: str) -> int:
    """How many bytes the row-group disk cache may use on this host.

    The old budget was a hardcoded 12 GiB, chosen for a 15 GB Colab disk. It
    is a *cache*, not a store: on a 500 GB box a 12 GiB cap holds four of the
    3 GiB row groups, so the cache thrashed (the measured eviction storm
    during the 4,000-step probe) and the loader re-fetched what it had just
    thrown away. Scale it with the disk instead, within bounds.
    """
    try:
        import shutil

        free = shutil.disk_usage(cache_dir).free
        by_disk = int(free * RG_CACHE_FREE_FRACTION)
        # never more than a quarter of the free space (the 12 GiB floor is for
        # a big disk; on a small one the budget shrinks with it)
        return max(min(256 * 2**20, free // 8),
                    min(RG_CACHE_MAX_BYTES, by_disk, free // 4))
    except Exception:
        return RG_CACHE_MIN_BYTES


def _enforce_rg_cache_limit(cache_dir: str | None, max_bytes: int | None = None) -> None:
    """Keep the rg_*.bin disk cache inside its budget (disk-derived by default).

    This is a CACHE: it may be dropped at any time, and nothing may depend on
    it surviving. The old fixed 12 GiB cap was sized for a Colab disk; on a
    big box it thrashed, on a small one it was the right size.
    """
    if not cache_dir or not os.path.isdir(cache_dir) or _in_modal():
        return
    budget = max_bytes if max_bytes is not None else _rg_cache_budget(cache_dir)
    try:
        import glob

        files = glob.glob(os.path.join(cache_dir, "rg_*.bin"))
        if not files:
            return
        files.sort(key=lambda p: os.path.getmtime(p))
        total = sum(os.path.getsize(p) for p in files)
        while total > budget and files:
            old = files.pop(0)
            try:
                sz = os.path.getsize(old)
                os.remove(old)
                total -= sz
                print(f"[stream] RG cache evicted {os.path.basename(old)} "
                      f"({sz/2**20:.0f}MiB, budget {budget/2**30:.0f}GiB)", flush=True)
            except Exception:
                pass
    except Exception:
        pass


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
) -> dict[str, tuple[str, int] | tuple[str, int, int]]:
    """emb key -> (shard_file, global_row_idx), footer+key-chunk only, cached."""
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key_index_path(cache_dir or ".", stream_order)
    if not rebuild:
        hit_t0 = time.time()
        index, ok = load_key_index(cache_path)
        if ok and len(index) > 0:
            print(f"[stream] key index loaded from cache ({len(index)} embeddings) <- {cache_path} in {time.time()-hit_t0:.1f}s", flush=True)
            return index
        else:
            # Verbose miss reason so Colab rebuild is explainable
            if cache_dir and not __import__("os").path.exists(cache_path):
                print(f"[stream] key index cache miss (no file at {cache_path}) — building", flush=True)
            elif not ok:
                print(f"[stream] key index cache invalid (version mismatch) at {cache_path} — rebuilding", flush=True)
    index: dict[str, tuple[str, int]] = {}
    t0 = time.time()
    import pyarrow.parquet as pq

    def _one_shard(sf: str) -> dict[str, tuple[str, int] | tuple[str, int, int]]:
        url = f"https://huggingface.co/datasets/{EMB_REPO}/resolve/main/{sf}"
        rs = RemoteShard(url, _remote_size(url))
        pf = pq.ParquetFile(rs)
        md = pf.metadata
        spans = _prefetch_key_spans(url, md)
        shard_index: dict[str, tuple[str, int] | tuple[str, int, int]] = {}
        row_cursor = 0
        for rgi in range(md.num_row_groups):
            lo, _ = rg_span(md, rgi, columns=("key", "n_vis"))
            rs._span = (lo, spans[rgi])
            tbl = pf.read_row_group(rgi, columns=["key", "n_vis"])
            keys_here = tbl.column("key").to_pylist()
            nvis_here = tbl.column("n_vis").to_pylist()
            g0 = md.row_group(rgi).num_rows
            for j, (k, nv) in enumerate(zip(keys_here, nvis_here)):
                if k in shard_index:
                    raise AssertionError(f"duplicate embedding key in shard {sf}: {k}")
                shard_index[k] = (sf, row_cursor + j, int(nv))
            row_cursor += g0
        return shard_index

    # Multi-channel: 8 shards in parallel — 213s → ~35s on Colab
    workers = min(8, len(stream_order))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_shard, sf): sf for sf in stream_order}
        done = 0
        for fut in as_completed(futs):
            sf = futs[fut]
            shard_index = fut.result()
            for k, v in shard_index.items():
                if k in index:
                    raise AssertionError(f"duplicate embedding key across shards: {k}")
                index[k] = v
            done += 1
            print(f"[stream]   key index {done}/{len(stream_order)} {Path(sf).name} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[stream] key index complete: {len(index)} embeddings in {time.time()-t0:.0f}s", flush=True)
    if cache_dir:
        save_key_index(index, cache_path)
        print(f"[stream] key index cached to {cache_path}", flush=True)
    return index


def build_epoch_plan(
    rows: list[dict],
    index: dict[str, tuple[str, int] | tuple[str, int, int]],
    sample_size: int,
    seed: int,
    excluded_shards: set[str] | None = None,
    bucket_by_n_vis: bool = True,
) -> dict[str, list[dict]]:
    """Cluster sampling with n_vis bucketing: shards are shuffled within bucket.

    When index carries n_vis (v3 sidecar), bucket boundaries from docs/DATA.md
    histogram (0-100/101-500/501-1000/1001-2000/2001-4900/4901+) keep batches
    size-homogeneous — the 66.8% 101-500 majority no longer pays 4900's cost.
    Backward-compatible: v2 index (no n_vis) falls back to plain shuffle."""
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
    # Bucket shards by median n_vis when v3 sidecar available
    if bucket_by_n_vis and any(len(v) == 3 for v in index.values()):
        def _median_n_vis(sf: str) -> int:
            nvis = [v[2] for k, v in index.items() if v[0] == sf and len(v) == 3]  # type: ignore[index]
            if not nvis:
                return 500  # fallback to dominant bucket center
            nvis.sort()
            return nvis[len(nvis)//2]
        buckets = [(0,100),(101,500),(501,1000),(1001,2000),(2001,4900),(4901,99999)]
        bucketed: dict[int, list[str]] = {i: [] for i in range(len(buckets))}
        for sf in sorted(by_shard_rows):
            med = _median_n_vis(sf)
            for i,(a,b) in enumerate(buckets):
                if a <= med <= b:
                    bucketed[i].append(sf)
                    break
        shard_files = []
        for i in range(len(buckets)):
            grp = bucketed[i]
            rng.shuffle(grp)
            shard_files.extend(grp)
    else:
        shard_files = sorted(by_shard_rows)
        rng.shuffle(shard_files)
    plan: dict[str, list[dict]] = {}
    budget = sample_size
    for sf in shard_files:
        if budget <= 0:
            break
        take = min(budget, len(by_shard_rows[sf]))
        if bucket_by_n_vis and any(len(v) == 3 for v in index.values()):
            # Sort within shard by n_vis so sequential batches are size-homogeneous
            def _row_nvis(t: tuple[int, dict]) -> int:
                emb = t[1].get("emb", "")
                loc = index.get(emb)
                return loc[2] if loc is not None and len(loc) == 3 else 500  # type: ignore[index]
            selected = sorted(by_shard_rows[sf], key=_row_nvis)[:take]
        else:
            selected = sorted(by_shard_rows[sf], key=lambda t: t[0])[:take]
        plan[sf] = [{**r, "_row": j} for j, r in selected]
        budget -= take
    print(f"[stream] cluster plan: {sum(len(v) for v in plan.values())} rows from {len(plan)} shards (bucketed={bucket_by_n_vis and any(len(v)==3 for v in index.values())})", flush=True)
    return plan


def group_rows_by_span(rows_here, g0: int) -> dict[int, dict[int, list[dict]]]:
    """Group plan rows by (row_group, position) -> the rows served for it.

    A position can hold **several** plan rows: the manifest asks several
    different questions about the same image (8,653 images carry 2-54 of them
    — chart/figure QA mostly). The previous mapping kept only the last row
    per position, so 45,393 of the 117,600 manifest rows (39%) were silently
    never served while `samples_seen` counted them.

    The list preserves plan order. The loader serves the questions of a group
    INTERLEAVED (question 1 of every position, then question 2 of every
    position, ...): serving them consecutively would put up to 54 questions
    about the SAME image in one batch, and the batch loss is a token-mean, so
    one image could dominate the gradient.
    """
    want: dict[int, dict[int, list[dict]]] = {}
    for r in rows_here:
        want.setdefault(r["_row"] // g0, {}).setdefault(r["_row"] % g0, []).append(r)
    return want


def interleaved_positions(want_rg: dict[int, list[dict]]) -> list[tuple[int, int]]:
    """(position, question index) in serve order, spread across the positions.

    Round-robin over the questions of a row group so that consecutive rows
    belong to DIFFERENT images.
    """
    positions = sorted(want_rg)
    if not positions:
        return []
    max_q = max(len(want_rg[p]) for p in positions)
    out: list[tuple[int, int]] = []
    for qi in range(max_q):
        for p in positions:
            if qi < len(want_rg[p]):
                out.append((p, qi))
    return out


def rows_in_span(want_rg: dict[int, list[dict]]) -> int:
    """How many plan rows a row group contributes to the stream."""
    return sum(len(v) for v in want_rg.values())


def span_is_skippable(emitted: int, n_rows: int, start_pos: int) -> bool:
    """True when EVERY row of this group is before the resume offset.

    Such a group must not be fetched: the 4,000-step probe spent 22.7 minutes
    fetching groups whose rows it then threw away.
    """
    return emitted + n_rows <= start_pos


class EmbStreamDataset(torch.utils.data.IterableDataset):
    """Shard-by-shard streaming with next-RG prefetch.

    A 1-worker background thread fetches the *next* row-group's
    ~400MiB span (4-way chunked Range, disk_cache-aware) while the
    current RG is yielded to the trainer. Hides ~30s fetch behind
    ~10s train steps; disk_cache hits skip the thread entirely.
    """

    PREFETCH_DEPTH = 1
    MAX_RG_ROWS = 128

    def __init__(
        self,
        plan: dict[str, list[dict]],
        stream_order: list[str],
        start_pos: int = 0,
        rg_cache_dir: str | None = None,
        vision_dim: int = VISION_DIM,
        local_shards: dict[str, str] | None = None,
    ):
        super().__init__()
        self.plan, self.order = plan, stream_order
        self.start_pos = start_pos
        self.rg_cache_dir = rg_cache_dir
        self.vision_dim = vision_dim
        # plan-shard name -> a parquet already on this disk. The val pack uses
        # this: one file, no fetch, no prefetch daemon. Rows still carry `_row`
        # because the reader indexes row groups by position either way.
        self.local_shards = local_shards or {}
        if rg_cache_dir:
            os.makedirs(rg_cache_dir, exist_ok=True)

    def __iter__(self):  # noqa: C901
        import pyarrow.parquet as pq

        emitted = 0
        # Phase 2 daemon: shard-level prefetch (whole-shard hf_transfer
        # ~1GiB/s). Not Modal-gated — the whole-shard path is open everywhere.
        shard_list = [s for s in self.order if self.plan.get(s)]
        shard_idx = {s: i for i, s in enumerate(shard_list)}
        next_shard_fut: _DaemonTask | None = None
        for sf in self.order:
            rows_here = self.plan.get(sf)
            if not rows_here:
                continue
            # Schedule next shard download while GPU trains current shard (pipelined 12min cold over 33h)
            if self.rg_cache_dir and sf in shard_idx:
                cur = shard_idx[sf]
                if cur + 1 < len(shard_list) and (next_shard_fut is None or next_shard_fut.done()):
                    nxt = shard_list[cur + 1]
                    nxt_path = self.local_shards.get(nxt) or _get_hf_shard_path(nxt, cache_dir=self.rg_cache_dir)
                    if nxt_path is None:
                        try:
                            next_shard_fut = _DaemonTask(_download_shard_hf_transfer, nxt, self.rg_cache_dir)
                        except Exception:
                            pass
            # Phase 2 fast path: whole-shard hf_transfer (~1 GiB/s) — local
            # parquet read, no Range. Not Modal-gated any more: the target is a
            # rented instance, and the Range path measures 1.5-28 MiB/s.
            local_path: str | None = self.local_shards.get(sf)
            if local_path is None:
                local_path = _get_hf_shard_path(sf, cache_dir=self.rg_cache_dir)
            if local_path is None:
                local_path = _download_shard_hf_transfer(sf, cache_dir=self.rg_cache_dir)
            if local_path and os.path.exists(local_path):
                pf = pq.ParquetFile(local_path)
                md = pf.metadata
                biggest = shard_row_group_size(md)
                assert biggest <= self.MAX_RG_ROWS, (
                    f"{sf}: row group {biggest} rows (~9 GiB) not streamable — exclude smoke shards"
                )
                g0 = md.row_group(0).num_rows or 64
                want = group_rows_by_span(rows_here, g0)
                for rgi in sorted(want):
                    # Resume skip (see the Range path): do not fetch groups
                    # whose rows are all before start_pos.
                    if span_is_skippable(emitted, rows_in_span(want[rgi]), self.start_pos):
                        emitted += rows_in_span(want[rgi])
                        continue
                    tbl = pq.ParquetFile(local_path).read_row_group(rgi, columns=["key", "n_vis", "vis_bytes"])
                    for j, qi in interleaved_positions(want[rgi]):
                        row = want[rgi][j][qi]
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
                        yield {
                            "vis": vis, "user": row["user"],
                            "assistant": row["assistant"], "g": row.get("g", "?"),
                            "grid_thw": row.get("grid_thw"),
                        }
                        emitted += 1
                _enforce_lru_cache(self.rg_cache_dir)
                if next_shard_fut is not None and next_shard_fut.done():
                    try:
                        next_shard_fut.value()
                    except Exception:
                        pass
                    next_shard_fut = None
                continue
            url = f"https://huggingface.co/datasets/{EMB_REPO}/resolve/main/{sf}"
            t_sf = time.time()
            rs = RemoteShard(url, _remote_size(url), disk_cache=self.rg_cache_dir)
            pf = pq.ParquetFile(rs)
            md = pf.metadata
            print(f"[stream] {Path(sf).name} metadata: {md.num_row_groups} RGs, "
                  f"{md.num_rows} rows ({time.time()-t_sf:.1f}s)", flush=True)
            biggest = shard_row_group_size(md)
            assert biggest <= self.MAX_RG_ROWS, (
                f"{sf}: row group {biggest} rows (~9 GiB) not streamable — exclude smoke shards"
            )
            g0 = md.row_group(0).num_rows or 64
            want = group_rows_by_span(rows_here, g0)
            rgis = sorted(want)

            # Prefetch the NEXT row group while the CURRENT one is being trained.
            #
            # The previous version submitted the fetch for rgis[idx+1] and then
            # immediately drained it *before* reading rgis[idx], so every
            # download sat between two training steps: the network was 100%
            # exposed (measured: the 4,000-step probe's 7,166 s gap was the
            # whole transfer time, and 46% of the GPU time went to the slow
            # groups). It also compared the next group's byte length against
            # the current group's size, so the "adopt" path never fired.
            #
            # Now: the fetch for rgis[idx+1] is submitted at the top of
            # iteration idx (so it overlaps the training of rgis[idx]), and
            # rgis[idx] is adopted by IDENTITY (lo, hi) at the top of
            # iteration idx+1. Same rows, same order, same bytes -- only WHEN
            # the transfer happens changes.
            pending: tuple[int, int, object] | None = None   # (lo, hi, task)

            def _start_prefetch(n_rgi: int):
                """Submit the fetch for n_rgi; returns (lo, hi, future) or None."""
                n_lo, n_hi = rg_span(md, n_rgi)
                probe = RemoteShard.__new__(RemoteShard)
                probe.url = url  # type: ignore[attr-defined]
                probe.disk_cache = self.rg_cache_dir  # type: ignore[attr-defined]
                n_cf = probe._cache_file(n_lo, n_hi)
                if os.path.exists(n_cf) and os.path.getsize(n_cf) == n_hi - n_lo:
                    return None            # already on disk -> the main thread reads it

                def _bg(n_lo=n_lo, n_hi=n_hi, n_rgi=n_rgi, sf=sf, size=rs.size):
                    rs2 = RemoteShard(url, size, disk_cache=self.rg_cache_dir)
                    t0 = time.time()
                    rs2.load_span(n_lo, n_hi)
                    dt = time.time() - t0
                    if dt > 2:
                        print(f"[stream] prefetched {Path(sf).name} rg{n_rgi} "
                              f"({(n_hi-n_lo)/2**20:.0f}MiB in {dt:.0f}s)", flush=True)
                    return rs2._span[1] if rs2._span else None

                return (n_lo, n_hi, _DaemonTask(_bg))

            try:
                for idx, rgi in enumerate(rgis):
                    # Resume skip: a row group whose rows are ALL before
                    # start_pos is never fetched. The old code fetched every
                    # group and then threw the rows away, which cost the
                    # 4,000-step probe 22.7 minutes before its first step.
                    if span_is_skippable(emitted, rows_in_span(want[rgi]), self.start_pos):
                        emitted += rows_in_span(want[rgi])
                        continue
                    lo, hi = rg_span(md, rgi)
                    t0 = time.time()
                    # 1) adopt the bytes prefetched for THIS group (identity, not length)
                    _adopted = None
                    if pending is not None and pending[0] == lo and pending[1] == hi:
                        _adopted = drain_row_group_prefetch(pending[2])
                        if _adopted is not None and isinstance(_adopted, bytes) \
                                and len(_adopted) == hi - lo:
                            rs._span = (lo, _adopted)
                            dt = time.time() - t0
                            print(f"[stream] adopted prefetched {Path(sf).name} rg{rgi} "
                                  f"({(hi-lo)/2**20:.0f}MiB, waited {dt:.0f}s)", flush=True)
                        else:
                            rs.load_span(lo, hi)
                            dt = time.time() - t0
                            if dt > 2:
                                print(f"[stream] streamed {Path(sf).name} rg{rgi} "
                                      f"({(hi-lo)/2**20:.0f}MiB in {dt:.0f}s)", flush=True)
                    else:
                        rs.load_span(lo, hi)
                        dt = time.time() - t0
                        if dt > 2:
                            print(f"[stream] streamed {Path(sf).name} rg{rgi} "
                                  f"({(hi-lo)/2**20:.0f}MiB in {dt:.0f}s)", flush=True)
                    # 2) start the NEXT group's fetch now, so it overlaps this group's training
                    pending = (_start_prefetch(rgis[idx + 1])
                              if (self.rg_cache_dir and idx + 1 < len(rgis)) else None)

                    tbl = pf.read_row_group(rgi, columns=["key", "n_vis", "vis_bytes"])
                    try:
                        for j, qi in interleaved_positions(want[rgi]):
                            row = want[rgi][j][qi]
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
                            yield {
                                "vis": vis, "user": row["user"],
                                "assistant": row["assistant"], "g": row.get("g", "?"),
                                "grid_thw": row.get("grid_thw"),
                            }
                            emitted += 1
                    finally:
                        del tbl
                        rs._span = None
            finally:
                # The prefetch runs on a daemon thread: an in-flight fetch can be
                # abandoned at exit (an executor worker would block the process).
                rs._span = None  # type: ignore[assignment]
            _enforce_lru_cache(self.rg_cache_dir)
            if next_shard_fut is not None and next_shard_fut.done():
                try:
                    next_shard_fut.value()
                except Exception:
                    pass
                next_shard_fut = None
        # The shard prefetch runs on a daemon thread: an in-flight download can
        # be abandoned at exit (a worker thread would block the process).
