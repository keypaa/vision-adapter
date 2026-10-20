"""Blob corpus: the format the rebuilt corpus uses, and its reader.

WHY THIS FORMAT (docs/DATA_PATH_REBUILD.md D2)
----------------------------------------------
The parquet shards are read in row groups, and a row group is the smallest unit
an HTTP Range request can serve. That is why serving 4,000 rows pulled 798 GiB
to deliver 250 GiB of useful embeddings: 3.19x overfetch, because a row group
holds rows the plan does not want.

The blob format removes the row group from the picture:

    emb_XXXX.bin   the rows of a shard, concatenated: row r occupies
                   [off_r, off_r + n_vis_r*4096*2) with
                   off_r = cumsum(n_vis*8192)
    emb_XXXX.idx   n_vis per row (uint16) -> 2.7 KB per 1,360-row shard
    rows.parquet   key -> (shard, row, n_vis, grid_thw) for the WHOLE corpus
    corpus.json    the shard list, row counts, byte counts, provenance

A row is then the unit of transfer: 2.9 MiB at the 383-token cap, and the
offsets are a prefix sum, so the index is tiny and the fetch is exact. The
plan can name any subset of rows, so "1% of the corpus" costs 1% of the bytes
and 1% of the time.

The reader is backend-agnostic on purpose (D3): a local directory and an HF
repo expose the SAME bytes, so a run on Modal and a run on a rental box serve
identical rows in an identical order. The only difference is where the bytes
come from.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORPUS_VERSION = 2
ROW_BYTES = 4096 * 2          # one 4096-dim bf16 token
DEFAULT_REPO = "keypa/vision-adapter-embeddings-v2"


@dataclass
class ShardInfo:
    name: str
    rows: int
    bytes: int


class BlobCorpus:
    """Read rows out of a blob corpus, locally or from an HF repo."""

    def __init__(self, root: str, key_map: dict[str, tuple[int, int, int]],
                 shards: list[ShardInfo], n_vis: dict[tuple[int, int], int],
                 grid: dict[tuple[int, int], list[int]] | None = None):
        self.root = root
        self.key_map = key_map
        self.shards = shards
        self._n_vis = n_vis
        self.grid = grid or {}
        self._local: dict[int, bytes] = {}     # shard idx -> the .idx bytes
        self._memmaps: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- open --
    @classmethod
    def open(cls, root: str, token: str | None = None) -> "BlobCorpus":
        """Open a corpus from a local directory or an HF dataset repo id."""
        if os.path.isdir(root):
            return cls._open_local(Path(root))
        return cls._open_repo(root, token)

    @classmethod
    def _open_local(cls, d: Path) -> "BlobCorpus":
        meta = json.loads((d / "corpus.json").read_text())
        shards = [ShardInfo(s["name"], s["rows"], s["bytes"]) for s in meta["shards"]]
        import pyarrow.parquet as pq

        tbl = pq.read_table(d / "rows.parquet")
        key_map: dict[str, tuple[int, int, int]] = {}
        n_vis: dict[tuple[int, int], int] = {}
        grid: dict[tuple[int, int], list[int]] = {}
        for key, sh, rw, nv, g in zip(
                tbl.column("key").to_pylist(),
                tbl.column("shard").to_pylist(),
                tbl.column("row").to_pylist(),
                tbl.column("n_vis").to_pylist(),
                tbl.column("grid_thw").to_pylist()):
            key_map[key] = (int(sh), int(rw), int(nv))
            n_vis[(int(sh), int(rw))] = int(nv)
            if g is not None:
                grid[(int(sh), int(rw))] = [int(x) for x in g]
        return cls(str(d), key_map, shards, n_vis, grid)

    @classmethod
    def _open_repo(cls, repo: str, token: str | None = None) -> "BlobCorpus":
        from huggingface_hub import hf_hub_download

        meta = json.loads(hf_hub_download(repo, "corpus.json", repo_type="dataset",
                                         token=token).read_text()
                          if isinstance(hf_hub_download(repo, "corpus.json",
                                                       repo_type="dataset", token=token),
                                        str)
                          else "{}")
        shards = [ShardInfo(s["name"], s["rows"], s["bytes"]) for s in meta["shards"]]
        import pyarrow.parquet as pq

        local = hf_hub_download(repo, "rows.parquet", repo_type="dataset", token=token)
        tbl = pq.read_table(local)
        key_map: dict[str, tuple[int, int, int]] = {}
        n_vis: dict[tuple[int, int], int] = {}
        grid: dict[tuple[int, int], list[int]] = {}
        for key, sh, rw, nv, g in zip(
                tbl.column("key").to_pylist(),
                tbl.column("shard").to_pylist(),
                tbl.column("row").to_pylist(),
                tbl.column("n_vis").to_pylist(),
                tbl.column("grid_thw").to_pylist()):
            key_map[key] = (int(sh), int(rw), int(nv))
            n_vis[(int(sh), int(rw))] = int(nv)
            if g is not None:
                grid[(int(sh), int(rw))] = [int(x) for x in g]
        return cls(repo, key_map, shards, n_vis, grid)

    # -------------------------------------------------------------- bytes --
    def _fetch(self, name: str) -> bytes:
        """The bytes of one auxiliary file (.idx), from disk or the repo."""
        if os.path.isdir(self.root):
            return (Path(self.root) / name).read_bytes()
        from huggingface_hub import hf_hub_download

        return open(hf_hub_download(self.root, name, repo_type="dataset"), "rb").read()

    def _row_offsets(self, shard: int) -> np.ndarray:
        """Byte offset of every row in a shard (cumsum of n_vis*ROW_BYTES)."""
        idx = self._local.get(shard)
        if idx is None:
            raw = self._fetch(self.shards[shard].name.replace(".bin", ".idx"))
            idx = np.frombuffer(raw, dtype=np.uint16)
            self._local[shard] = idx
        return np.concatenate(([0], np.cumsum(idx.astype(np.int64) * ROW_BYTES)))

    def shard_bytes(self, shard: int) -> bytes:
        """The whole .bin of a shard (used by the local path and by tests)."""
        if os.path.isdir(self.root):
            return (Path(self.root) / self.shards[shard].name).read_bytes()
        from huggingface_hub import hf_hub_download

        return open(hf_hub_download(self.root, self.shards[shard].name,
                                    repo_type="dataset"), "rb").read()

    def read_range(self, shard: int, offset: int, length: int) -> bytes:
        """A byte range of a shard's .bin — the unit of transfer.

        Local: a seek+read. Repo: ONE HTTP Range request. This is why the
        format pays: the loader asks for the bytes of the rows it is about to
        serve, not for the shard.
        """
        if length <= 0:
            return b""
        if os.path.isdir(self.root):
            with open(Path(self.root) / self.shards[shard].name, "rb") as f:
                f.seek(offset)
                return f.read(length)
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(self.root, self.shards[shard].name, repo_type="dataset")
        req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
        tok = os.environ.get("HF_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def row_bytes(self, shard: int, row: int) -> bytes:
        """The exact bytes of one row (2.9 MiB at the 383-token cap)."""
        off = self._row_offsets(shard)
        lo, hi = int(off[row]), int(off[row + 1])
        blob = self.shard_bytes(shard)
        return blob[lo:hi]

    def row_range(self, shard: int, row: int) -> tuple[int, int]:
        """(offset, length) of a row inside its shard's .bin — for Range fetches."""
        off = self._row_offsets(shard)
        return int(off[row]), int(off[row + 1] - off[row])

    def rows_range(self, shard: int, rows: list[int]) -> tuple[int, int]:
        """(offset, length) covering a set of rows — one coalesced Range request."""
        off = self._row_offsets(shard)
        lo = int(off[min(rows)])
        hi = int(off[max(rows) + 1])
        return lo, hi - lo

    # --------------------------------------------------------------- rows --
    def lookup(self, key: str) -> tuple[int, int, int]:
        """key -> (shard, row, n_vis)."""
        try:
            return self.key_map[key]
        except KeyError:
            raise KeyError(f"key not in the corpus: {key}") from None

    def n_vis(self, shard: int, row: int) -> int:
        return self._n_vis[(shard, row)]

    def grid_thw(self, shard: int, row: int) -> list[int] | None:
        return self.grid.get((shard, row))

    def row_tensor(self, shard: int, row: int, dtype=np.float32):
        """One row as a numpy array [n_vis, 4096] (bf16 bytes)."""
        nv = self.n_vis(shard, row)
        raw = self.row_bytes(shard, row)
        arr = np.frombuffer(bytearray(raw), dtype=np.uint16).reshape(nv, 4096)
        return arr.view(np.float16) if dtype is np.float16 else arr

    def __len__(self) -> int:
        return len(self.key_map)

    def __contains__(self, key: str) -> bool:
        return key in self.key_map


def rows_of_shard(corpus: BlobCorpus, shard: int) -> list[int]:
    """The row indices of a shard, in order."""
    return [r for (s, r) in corpus._n_vis if s == shard]


def build_row_plan(corpus: BlobCorpus, keys: list[str], sample_size: int,
                   seed: int = 0) -> list[tuple[int, int]]:
    """A ROW-granular plan: which (shard, row) pairs to serve, in order.

    The parquet plan is shard-granular (build_epoch_plan takes whole shards),
    so a small run still pays for whole shards. With the blob format the plan
    can name individual rows, and the fetch is exactly those rows' bytes:
    "1% of the corpus" costs 1% of the bytes and 1% of the time.

    The order is deterministic: the keys are sorted, shuffled with `seed`, and
    then regrouped by shard so that consecutive rows of a shard are fetched
    together (one coalesced range per group).
    """
    import random

    wanted = [k for k in keys if k in corpus]
    if sample_size and sample_size < len(wanted):
        rng = random.Random(seed)
        rng.shuffle(wanted)
        wanted = sorted(wanted[:sample_size])
    else:
        wanted = sorted(wanted)
    plan: list[tuple[int, int]] = []
    by_shard: dict[int, list[int]] = {}
    for k in wanted:
        sh, rw, _nv = corpus.lookup(k)
        by_shard.setdefault(sh, []).append(rw)
    for sh in sorted(by_shard):
        for rw in sorted(by_shard[sh]):
            plan.append((sh, rw))
    return plan


class BlobStreamDataset:
    """Serve a row plan out of a blob corpus, one coalesced range per group.

    Backend-agnostic by construction: a local directory and an HF repo serve
    the same bytes in the same order (D3), so a run on Modal and a run on a
    rental box are the same experiment.
    """

    def __init__(self, corpus: BlobCorpus, plan: list[tuple[int, int]],
                 start_pos: int = 0, group: int = 16):
        self.corpus = corpus
        self.plan = plan
        self.start_pos = start_pos
        self.group = group          # rows per coalesced fetch (a batch)

    def __iter__(self):
        import torch

        emitted = 0
        i = 0
        n = len(self.plan)
        while i < n:
            chunk = self.plan[i:i + self.group]
            if not chunk:
                break
            shard = chunk[0][0]
            same = [c for c in chunk if c[0] == shard]
            rows = [r for _s, r in same]
            off, length = self.corpus.rows_range(shard, rows)
            blob = self.corpus.read_range(shard, off, length)
            # slice each row out of the coalesced blob
            for (s, r) in same:
                nv = self.corpus.n_vis(s, r)
                start = self.corpus.row_range(s, r)[0] - off
                raw = blob[start:start + nv * ROW_BYTES]
                if emitted < self.start_pos:
                    emitted += 1
                    continue
                # bytearray (not bytes) so the buffer is writable: torch.from_numpy
                # warns on a read-only array.
                arr = np.frombuffer(bytearray(raw), dtype=np.uint16).reshape(nv, 4096)
                yield torch.from_numpy(arr).view(torch.bfloat16)
                emitted += 1
            i += len(same)


def write_corpus_meta(path: Path, shards: list[ShardInfo], cap_patches: int,
                      source: str, extra: dict | None = None) -> dict:
    """The corpus.json manifest."""
    meta = {
        "version": CORPUS_VERSION,
        "format": "blob+idx",
        "row_bytes": ROW_BYTES,
        "cap_patches": cap_patches,
        "source": source,
        "total_rows": sum(s.rows for s in shards),
        "total_bytes": sum(s.bytes for s in shards),
        "shards": [{"name": s.name, "rows": s.rows, "bytes": s.bytes} for s in shards],
    }
    if extra:
        meta.update(extra)
    path.write_text(json.dumps(meta, indent=2))
    return meta
