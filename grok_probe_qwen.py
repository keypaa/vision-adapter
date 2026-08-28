#!/usr/bin/env python3
"""grok_probe_qwen.py — Rung 1 of the vision-adapter validation ladder.

Trains ONLY the HourglassProjector against a small text-only Qwen3.5 LLM
(2B default), consuming the EXISTING precomputed MoonViT-V2 embeddings
(`keypa/vision-adapter-embeddings`, 103 parquet shards) and the EXISTING train
manifest (`keypa/vision-adapter-manifests`), subsampled to fit a free Colab
session. Purpose: prove the recipe end-to-end on cheap hardware AND collect a
real grok-window data point (at what samples_seen does loss collapse, if ever).

Mirrors modal_train.py's contracts (deviations are printed at startup and
listed in GROK_PROBE.md):

    sequence layout per example (identical to modal_train.make_collate):
        [BOS] [img_emb x n_vis] [user text tokens] [answer tokens] [EOS]
      - image embeddings spliced at positions [1 : 1+n_vis]. Here we pass
        inputs_embeds ONLY: Qwen3.5 raises ValueError on input_ids+inputs_embeds
        together ("specify exactly one"), and unlike DeepSeek-V4 there is no
        hash-MoE id lookup forcing ids to stay the model input. Base rows are
        emb_lookup(ids); the visual span is overwritten with projected cached
        embeddings so grads flow to the projector. Qwen DOES have an image
        sentinel (<|image_pad|>, id from config.image_token_id) but it is not
        needed: placeholder-id positions stay masked out of the loss (-100)
        exactly as in modal_train.
      - labels -100 everywhere except answer + EOS. BOS, image span, user
        prompt, right-pad never contribute to loss. Pinned by invariant checks
        ported from test_train_collate.py (asserted on the first batch).

Precision: bf16/fp32 only — NO quantization anywhere in this probe; the point
is a clean grok signal uncontaminated by FP8 residency questions.

Telemetry contract (non-negotiable): every print flush=True; timed startup
phases; per-step JSONL {step, loss, ema_loss, gnorm, lr, tok_s, samples_seen,
elapsed_s} to probe_log.jsonl; probe_curves.png re-rendered every CHART_EVERY
steps with x-axes in BOTH steps and samples_seen; explicit flat-loss plateau
banner naming the Baseten reference; spike alert when EMA jumps >2x;
checkpoint every SAVE_EVERY steps (resumable: rerun with --resume).

Run:
    python grok_probe_qwen.py --model qwen2b --sample-size 20000 \
        --batch-size 8 --max-steps 3000 --resume
"""
from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import statistics
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from vision_adapter.config import colab_probe_config as _colab_probe_config, config_header as _config_header  # single source of truth
from vision_adapter.core import (  # single shared training core — see vision_adapter/core.py
    HourglassProjector as _CoreHourglass,
    ProbeMonitor as _CoreProbeMonitor,
    TrainMonitor as _CoreTrainMonitor,
    check_collate_invariants as _core_check,
    embeds_for as _core_embeds,
    lr_at as _core_lr_at,
    make_collate as _core_make_collate,
    render_curves as _core_render_curves,
    render_train_curves as _core_render_train_curves,
    train_step_qwen as _core_train_step_qwen,
    visual_inject as _CoreVisualInject,
)

_T0 = time.time()

# ----------------------------- constants ------------------------------------

MODELS = {
    "qwen2b": "Qwen/Qwen3.5-2B",   # hidden 2048, 24 layers — default (free Colab)
    "qwen4b": "Qwen/Qwen3.5-4B",   # hidden 2560, 32 layers
}

def _get_hf_token(cli_token: str | None = None) -> str | None:
    """Resolve HF token. Priority: CLI arg > env > Colab userdata.

    Important Colab gotcha: `!python script.py` runs in a SUBPROCESS that
    does NOT inherit `google.colab.userdata` — you must explicitly forward
    the secret in the notebook before launching:

        from google.colab import userdata; import os
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

    This function handles all three paths so the script works both ways."""
    if cli_token:
        return cli_token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata  # only available inside Colab kernel
        token = userdata.get("HF_TOKEN")
        if token:
            return str(token)
    except Exception:
        pass
    return None


# resolved early; may be overridden after parse_args if --hf-token is given
HF_TOKEN: str | None = _get_hf_token()

EMB_REPO = "keypa/vision-adapter-embeddings"
MANIFEST_REPO = "keypa/vision-adapter-manifests"
MANIFEST_FILE = "train_manifest.jsonl"
LOG_FILE = "probe_log.jsonl"
CURVES_PNG = "probe_curves.png"
CKPT_DIR = "probe_ckpts"
FINAL_PATH = "projector_probe_final.safetensors"
KEY_INDEX_CACHE = "key_index_cache.json"       # in cache_dir

# Single source of truth — see vision_adapter/config.py (replaces the 3×
# scattered LR/BATCH_SIZE/MAX_SEQ_LEN/WARMUP at grok_probe:104 /
# modal_probe:65 / modal_train:49). Module-level aliases kept for
# test compatibility (test_grok_probe_smoke imports gp.LR etc.).
_CFG = _colab_probe_config()
VISION_DIM = _CFG.vision_dim
LR = _CFG.lr
WARMUP_STEPS = _CFG.warmup_steps
GRAD_CLIP = _CFG.grad_clip
MAX_SEQ_LEN = _CFG.max_seq_len
EPOCHS = _CFG.epochs
CHART_EVERY = _CFG.chart_every
SAVE_EVERY = _CFG.save_every
STATUS_EVERY = _CFG.status_every
SAMPLES_PER_BASETEN_GROK = _CFG.samples_per_baseten_grok
PLATEAU_CHECK_EVERY = _CFG.plateau_check_every
PLATEAU_WINDOW = _CFG.plateau_window
PLATEAU_REL_TOL = _CFG.plateau_rel_tol
EMA_BETA = _CFG.ema_beta
SPIKE_FACTOR = _CFG.spike_factor
SPIKE_WINDOW = _CFG.spike_window
SPIKE_MIN_HISTORY = _CFG.spike_min_history


def _phase(msg: str) -> None:
    print(f"[probe] +{time.time() - _T0:6.1f}s  {msg}", flush=True)


# Python 3.14+ defaults to 'forkserver' on POSIX, which can't pickle closures
# like our `collate`. Force 'fork' so DataLoader workers work.
try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass


def enforce_ram_cap_gib(cap_gib: float) -> None:
    """Hard self-limit so a runaway probe can never take the host down.

    Sets RLIMIT_AS (address space) for THIS process and any forked children.
    On breach, allocations fail with MemoryError instead of swapping the box
    into a coma. Must be called BEFORE big allocations (model load etc.)."""
    import resource
    limit = int(cap_gib * 2 ** 30)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        print(f"[probe] RAM cap enforced at {cap_gib:.0f} GiB "
              f"(RLIMIT_AS; allocations beyond it raise instead of swap)",
              flush=True)
    except Exception as e:
        print(f"[probe] WARNING: could not set RLIMIT_AS ({e}) — running "
              f"without a hard cap", flush=True)


# ----------------------------- projector ------------------------------------
# Canonical implementation in vision_adapter/core.py
HourglassProjector = _CoreHourglass


# ----------------------------- data -----------------------------------------


def fetch_manifest(cache_dir: str | None = None) -> list[dict]:
    """Download + parse train_manifest.jsonl ({emb, user, assistant, g}).

    Tolerates a leading manifest_header row (vision_adapter/manifest.py) — skipped."""
    from huggingface_hub import hf_hub_download
    kw = dict(local_dir=cache_dir) if cache_dir else {}
    local = hf_hub_download(MANIFEST_REPO, MANIFEST_FILE, repo_type="dataset", **kw)
    rows = []
    with open(local) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if obj.get("type") == "manifest_header":
                    continue
                rows.append(obj)
    return rows


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def list_shards() -> list[str]:
    """The 103 parquet shard paths inside EMB_REPO, sorted (packer order)."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN) if HF_TOKEN else HfApi()
    files = api.list_repo_files(EMB_REPO, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def _remote_size(url: str) -> int:
    import urllib.request
    # HEAD must NOT carry a Range header — CDN returns 400 otherwise
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
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s, 2s
    raise last_err


class RemoteShard(io.RawIOBase):
    """A remote parquet shard made locally seekable via HTTPS Range requests.

    Verified against the live repo before writing this:
      - shards are 22 row groups of 64 rows (~400 MiB each), NOT one giant
        group — so row-group-granular streaming works;
      - pyarrow only ever seeks to the footer and to the exact column-chunk
        spans of the row groups it reads, so a seekable file-like that serves
        [pre-fetched RG span][footer] lets us stream ONE row group at a time;
      - a ~400 MiB span fetched with 4 parallel streams takes ~8 s
        (~55 MiB/s) — fully hidden behind training steps once pipelined;
      - RAM peak = one decoded row-group table + one in-flight fetch ~= 1 GiB.
    """

    FOOTER_BYTES = 64 * 2 ** 10
    FETCH_CHUNK = 16 * 2 ** 20
    N_STREAMS = 4

    def __init__(self, url: str, size: int, disk_cache: str | None = None):
        super().__init__()
        self.url, self.size = url, size
        self.pos = 0
        self.disk_cache = disk_cache          # dir for per-RG span files
        self.footer = _fetch_range(url, size - self.FOOTER_BYTES, size - 1)
        self._span: tuple[int, bytes] | None = None   # (lo, blob) in RAM

    def _cache_file(self, lo: int, hi: int) -> str:
        import hashlib
        h = hashlib.sha1(f"{self.url}:{lo}:{hi}".encode()).hexdigest()[:20]
        return os.path.join(self.disk_cache, f"rg_{h}.bin")

    def load_span(self, lo: int, hi: int) -> None:
        """Pre-fetch [lo, hi) with parallel streams; disk-cached so epoch 2+
        and resumed runs never re-stream a group."""
        if self._span is not None and self._span[0] == lo:
            return
        if self.disk_cache:
            cf = self._cache_file(lo, hi)
            if os.path.exists(cf):
                with open(cf, "rb") as f:
                    blob = f.read()
                if len(blob) == hi - lo:      # complete file wins over partial
                    self._span = (lo, blob)
                    return
                os.remove(cf)                 # incomplete -> refetch
        bounds = [(st, min(st + self.FETCH_CHUNK, hi))
                  for st in range(lo, hi, self.FETCH_CHUNK)]
        with ThreadPoolExecutor(self.N_STREAMS) as ex:
            futs = {ex.submit(_fetch_range, self.url, st, en - 1): (st, en)
                    for st, en in bounds}
            got = {st: fut.result() for fut, (st, en) in futs.items()}
        blob = b"".join(got[st] for st, _ in bounds)
        if self.disk_cache and len(blob) == hi - lo:
            tmp = self._cache_file(lo, hi) + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(blob)
                os.replace(tmp, self._cache_file(lo, hi))
            except Exception:
                pass                          # cache write is best-effort
        self._span = (lo, blob)

    # --- file-like protocol (only footer + loaded span are served) ---
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.size + off if whence == 2 else self.pos)
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def read(self, n=-1):
        if n == -1:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        if self.pos >= self.size - len(self.footer):
            off = self.pos - (self.size - len(self.footer))
            chunk = self.footer[off: off + (end - self.pos)]
        elif self._span is not None:
            lo, blob = self._span
            if lo <= self.pos < lo + len(blob):
                chunk = blob[self.pos - lo: self.pos - lo + (end - self.pos)]
            else:
                raise OSError(f"read at {self.pos} outside pre-fetched span "
                              f"[{lo}, {lo + len(blob)}) — call load_span first")
        else:
            raise OSError("no row group pre-fetched")
        self.pos += len(chunk)
        return chunk


def rg_span(md, rg_idx: int, columns: tuple[str, ...] | None = None) -> tuple[int, int]:
    """[start, end) byte range covering a row group's column chunks.

    columns=None -> all columns (the training-read span); pass ("key",) to
    fetch only the key column when indexing (~KBs instead of ~400 MiB).
    """
    rg = md.row_group(rg_idx)
    names = [md.row_group(rg_idx).column(c).path_in_schema
             for c in range(rg.num_columns)]
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
    """Rows per row group. Two packer generations exist in the repo: the two
    Modal smoke shards (emb_0000/0001) have ONE 1360-row group; local_pack
    shards have 22 x 64-row groups. The streaming design needs SMALL groups —
    refuse giant ones rather than OOM Colab RAM on a 9-GiB span."""
    biggest = max(md.row_group(g).num_rows for g in range(md.num_row_groups))
    return biggest


def _prefetch_key_spans(url: str, md) -> dict[int, bytes]:
    """Fetch all key-column spans of a shard in parallel; returns rgi -> blob."""
    spans = [(rgi, *rg_span(md, rgi, columns=("key",)))
             for rgi in range(md.num_row_groups)]
    with ThreadPoolExecutor(max_workers=min(12, len(spans))) as ex:
        futs = {ex.submit(_fetch_range, url, lo, hi - 1): rgi
                for rgi, lo, hi in spans}
        return {futs[fut]: fut.result() for fut in as_completed(futs)}


def _cache_key_index_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, KEY_INDEX_CACHE)


def save_key_index(index: dict[str, tuple[str, int]], path: str) -> None:
    payload = {
        # v2: v1 cached per-row-group indices (parallel-build bug); rows must
        # be GLOBAL within the shard. Bump forces old caches to rebuild.
        "version": 2,
        "keys": {k: {"shard": s, "row": r} for k, (s, r) in index.items()},
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def load_key_index(path: str) -> tuple[dict[str, tuple[str, int]], bool]:
    """Load cached key index; returns (index, valid). valid=False if missing/corrupt."""
    if not os.path.exists(path):
        return {}, False
    try:
        with open(path) as f:
            payload = json.load(f)
        if payload.get("version") != 2 or not isinstance(payload.get("keys"), dict):
            return {}, False          # stale v1 cache -> caller rebuilds
        return {k: (v["shard"], v["row"]) for k, v in payload["keys"].items()}, True
    except Exception:
        return {}, False


def build_key_index(stream_order: list[str], cache_dir: str | None = None,
                    rebuild: bool = False) -> dict[str, tuple[str, int]]:
    """emb key -> (shard_file, global_row_idx), footer+key-chunk reads only.

    Each shard costs one footer fetch + 22 parallel key-chunk fetches (one per
    row group) — NOT the vis_bytes body (~400 MiB/group). Rows are numbered
    GLOBALLY within the shard (rg0 rows 0..63, rg1 rows 64..127, ...) matching
    EmbStreamDataset's `_row // g0` / `_row % g0` split. Cached to disk."""
    cache_path = _cache_key_index_path(cache_dir or ".")
    if not rebuild:
        index, ok = load_key_index(cache_path)
        if ok and len(index) > 0:
            print(f"[probe] key index loaded from cache ({len(index)} embeddings)",
                  flush=True)
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
        row_cursor = 0                            # global row within THIS shard
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
            row_cursor += g0                      # advance by the FULL group
        print(f"[probe]   key index {i + 1}/{len(stream_order)} "
              f"{os.path.basename(sf)} ({time.time() - t0:.0f}s)",
              flush=True)
    print(f"[probe] key index complete: {len(index)} embeddings in "
          f"{time.time() - t0:.0f}s", flush=True)
    if cache_dir:
        save_key_index(index, cache_path)
        print(f"[probe] key index cached to {cache_path}", flush=True)
    return index


def build_epoch_plan(rows: list[dict], index: dict[str, tuple[str, int]],
                     sample_size: int, seed: int,
                     excluded_shards: set[str] | None = None) -> dict[str, list[dict]]:
    """CLUSTER sampling: seed-shuffle SHARDS, take whole shards until the row
    budget is met.

    Why not global-random rows: a 20k-of-117k random slice touches all 101
    shards, so nearly every ~400-MiB streamed group yields ~1 wanted row —
    measured on Colab at ~400 MiB of HTTP per single training sample (hours
    of streaming per epoch). Whole-shard selection downloads each group once
    and every row inside it trains. Statistical concession: rows within a
    shard share packer order (grouped by source), but shard choice is
    seed-random over the full corpus; grok-window math is per samples_seen
    and unaffected. Deterministic given --seed => --resume replays exactly.
    Excluded shards (old giant-row-group smoke shards) are never picked.
    """
    rng = random.Random(seed)
    excluded = {f"datasets/{s}" if "/" in s and not s.startswith("data/")
                else s for s in (excluded_shards or set())}
    # rows grouped per shard first (only shards we can actually stream)
    by_shard_rows: dict[str, list[tuple[int, dict]]] = {}
    for r in rows:
        loc = index.get(r.get("emb"))
        if loc is None or loc[0] in excluded:
            continue
        by_shard_rows.setdefault(loc[0], []).append((loc[1], r))
    n_available = sum(len(v) for v in by_shard_rows.values())
    dropped = len(rows) - n_available
    if dropped:
        print(f"[probe] NOTE: {dropped} manifest rows skipped "
              f"(no embedding or in excluded shards)", flush=True)

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
    print(f"[probe] cluster plan: {sum(len(v) for v in plan.values())} rows "
          f"from {len(plan)} whole shards (~{len(plan)} x 9 GiB streamed once)",
          flush=True)
    return plan


class EmbStreamDataset(torch.utils.data.IterableDataset):
    """Yields planned rows shard-by-shard, streaming each shard's row groups
    over HTTPS (RemoteShard) with a background prefetcher on the NEXT group.

    Reconstruction pinned by test_local_pack.py:
        np.frombuffer(vis_bytes, uint8) -> torch.bfloat16 -> reshape(-1,4096).
    start_pos fast-forwards resumed runs deterministically.
    """

    PREFETCH_DEPTH = 2          # row groups kept fetched ahead of the reader

    def __init__(self, plan: dict[str, list[dict]], stream_order: list[str],
                 start_pos: int = 0, rg_cache_dir: str | None = None):
        super().__init__()
        self.plan, self.order = plan, stream_order
        self.start_pos = start_pos
        self.rg_cache_dir = rg_cache_dir
        if rg_cache_dir:
            os.makedirs(rg_cache_dir, exist_ok=True)

    MAX_RG_ROWS = 128           # refuse giant row groups (RAM guard)

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
            assert biggest <= self.MAX_RG_ROWS, \
                (f"{sf}: row group of {biggest} rows (~9 GiB span) would blow "
                 f"the RAM budget — only local_pack-generation shards "
                 f"(64-row groups) are streamable. emb_0000/0001 are old "
                 f"Modal smoke shards; exclude them from the plan.")
            g0 = md.row_group(0).num_rows or 64
            want: dict[int, dict[int, dict]] = {}   # rgi -> {local_j -> row}
            for r in rows_here:
                want.setdefault(r["_row"] // g0, {})[r["_row"] % g0] = r
            for rgi in sorted(want):
                lo, hi = rg_span(md, rgi)
                t0 = time.time()
                rs.load_span(lo, hi)
                dt = time.time() - t0
                if dt > 2:
                    print(f"[probe] streamed {os.path.basename(sf)} rg{rgi} "
                          f"({(hi - lo) / 2 ** 20:.0f}MiB in {dt:.0f}s)",
                          flush=True)
                tbl = pf.read_row_group(rgi,
                                        columns=["key", "n_vis", "vis_bytes"])
                try:
                    for j in sorted(want[rgi]):
                        row = want[rgi][j]
                        if emitted < self.start_pos:
                            emitted += 1
                            continue
                        vb = tbl.column("vis_bytes")[j].as_py()
                        nv = int(tbl.column("n_vis")[j].as_py())
                        # copy once into a writable buffer, then free ASAP
                        buf = bytearray(vb)
                        del vb
                        vis = (torch.from_numpy(np.frombuffer(buf, dtype=np.uint8))
                               .view(torch.bfloat16)
                               .reshape(-1, VISION_DIM)
                               .float())
                        del buf
                        assert vis.shape == (nv, VISION_DIM), \
                            f"embedding schema mismatch for {row['emb']}: " \
                            f"n_vis={nv} vs {tuple(vis.shape)}"
                        yield {"vis": vis, "user": row["user"],
                               "assistant": row["assistant"],
                               "g": row.get("g", "?")}
                        emitted += 1
                finally:
                    # release the 404 MiB blob + pyarrow table before next RG
                    del tbl
                    rs._span = None

# ----------------------------- telemetry ------------------------------------
# Canonical implementations in vision_adapter/core.py
def make_collate(tok, pad_id: int, max_len: int = MAX_SEQ_LEN):
    return _core_make_collate(tok, pad_id, max_len=max_len, vision_dim=VISION_DIM)

check_collate_invariants = _core_check  # kept but now re-exports core

embeds_for = _core_embeds  # inputs_embeds-only injection; see core.py



# Canonical in core.py (same banner text; reads _CFG for window/thresholds)
ProbeMonitor = _CoreProbeMonitor


# Canonical implementations in vision_adapter/core.py (same curves, LR, selective loss)
render_curves = _core_render_curves
lr_at = _core_lr_at
train_step = _core_train_step_qwen


def save_projector(proj, opt, step: int, path: str) -> None:
    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().contiguous() for k, v in proj.state_dict().items()}
    save_file(sd, path)
    if opt is not None:
        torch.save({"opt": opt.state_dict(), "step": step},
                   os.path.splitext(path)[0] + ".opt.pt")


# ----------------------- crash-resilient HF bundle ---------------------------

CKPT_REPO = "keypa/vision-adapter-grok-probe"
BUNDLE_FILES = ("latest.safetensors", "latest.opt.pt",
                LOG_FILE, CURVES_PNG)


def push_bundle(out_dir: str, ckpts: str, step: int) -> bool:
    """Upload the resumable bundle to HF so a dead Colab VM loses nothing but
    the steps since the last SAVE_EVERY. Best-effort: no token / no net must
    never kill training (the local copies remain the source of truth)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        if api.whoami()["auth"]["type"] != "accessToken":
            pass  # fine — any authed identity may push; only absence is fatal
        api.create_repo(CKPT_REPO, repo_type="model", exist_ok=True)
        paths = [os.path.join(ckpts, "latest.safetensors"),
                 os.path.join(ckpts, "latest.opt.pt"),
                 os.path.join(out_dir, LOG_FILE),
                 os.path.join(out_dir, CURVES_PNG)]
        for p in paths:
            if os.path.exists(p):
                api.upload_file(path_or_fileobj=p,
                                path_in_repo=os.path.basename(p),
                                repo_id=CKPT_REPO, repo_type="model",
                                commit_message=f"probe step {step}")
        print(f"[probe] bundle pushed to hf.co/{CKPT_REPO} @ step {step}",
              flush=True)
        return True
    except Exception as e:
        print(f"[probe] WARNING: HF bundle push skipped ({type(e).__name__}: "
              f"{e}) — checkpoints stay local-only this cycle", flush=True)
        return False


def pull_bundle(cache_dir: str) -> bool:
    """Fetch the last-pushed bundle into cache_dir; True if a usable checkpoint
    pair came down (caller resumes from it)."""
    try:
        from huggingface_hub import hf_hub_download
        got_ckpt = False
        for fn in BUNDLE_FILES:
            try:
                hf_hub_download(CKPT_REPO, fn, repo_type="model",
                                local_dir=cache_dir)
                got_ckpt |= fn.endswith((".safetensors", ".opt.pt"))
            except Exception:
                pass  # missing file in repo (e.g. no PNG yet on first push)
        return got_ckpt
    except Exception as e:
        print(f"[probe] no remote bundle found ({type(e).__name__})", flush=True)
        return False


# ----------------------------- CLI ------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="small-model grok-probe trainer "
                                             "(Qwen3.5 + cached MoonViT embeddings)")
    ap.add_argument("--model", choices=list(MODELS), default="qwen2b")
    ap.add_argument("--sample-size", type=int, default=20000,
                    help="random seeded slice of the train manifest")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto",
                    help="'auto' = bf16 on Ampere+, fp32 elsewhere (e.g. T4)")
    ap.add_argument("--cache-dir", default="emb_cache",
                    help="local dir for HF downloads (manifest, resume bundle, key index cache)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--resume", action="store_true",
                    help="resume from probe_ckpts/ (local first, then the HF "
                         "bundle pushed by a previous — possibly crashed run)")
    ap.add_argument("--hf-token", default=None,
                    help="HF read token. Colab gotcha: `!python grok_probe_qwen.py` "
                         "runs in a SUBPROCESS that does not see "
                         "google.colab.userdata, so forward the Secret into env "
                         "BEFORE launching: "
                         "from google.colab import userdata; import os; "
                         "os.environ['HF_TOKEN']=userdata.get('HF_TOKEN')  "
                         "or pass it directly: --hf-token HF_xxx. "
                         "Env fallbacks: HF_TOKEN / HUGGING_FACE_HUB_TOKEN.")
    ap.add_argument("--limit-layers", type=int, default=None,
                    help="keep only the first N backbone layers (smoke tests)")
    ap.add_argument("--ram-cap-gib", type=float, default=None,
                    help="hard RLIMIT_AS self-cap in GiB (e.g. 12); process "
                         "raises MemoryError beyond it instead of swapping "
                         "the host. Recommended on shared/local machines.")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="ignore cached key index and rebuild from scratch")
    return ap.parse_args()


def load_backbone(model_key: str, limit_layers: int | None, device: str, dtype):
    """Frozen backbone, plain precision — NO quantization anywhere in the probe."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = MODELS[model_key]
    _phase(f"tokenizer {repo} ...")
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    _phase(f"loading backbone {repo} ({dtype}, no quantization) — "
           f"this is the long part on a cold VM ...")
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=dtype, low_cpu_mem_usage=True, device_map=device)
    if limit_layers is not None:
        model.model.layers = model.model.layers[:limit_layers]
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    # train(), NOT eval(): gradient-checkpointing gates on self.training
    # (same gotcha as modal_train; safe here — no stochastic layers).
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = getattr(model.config, "text_config", model.config)
    _phase(f"backbone ready: hidden_size={cfg.hidden_size} "
           f"layers={len(model.model.layers)} | bos={tok.bos_token_id} "
           f"eos={tok.eos_token_id} pad={tok.pad_token_id} "
           f"image_sentinel={getattr(model.config, 'image_token_id', None)} (unused)")
    return tok, model, int(cfg.hidden_size)


def main():
    args = parse_args()
    # allow --hf-token to override module-level resolution
    global HF_TOKEN
    if args.hf_token is not None:
        HF_TOKEN = args.hf_token
        os.environ["HF_TOKEN"] = HF_TOKEN
    print(f"[probe] hf token: {'present (auth enabled)' if HF_TOKEN else 'absent (anonymous)'}",
          flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, LOG_FILE)
    curves_path = os.path.join(args.out_dir, CURVES_PNG)
    ckpts = os.path.join(args.out_dir, CKPT_DIR)
    os.makedirs(ckpts, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    props = ""
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        cc = p.major * 10 + p.minor
        # auto: bf16 on Ampere+ (cc>=80) — weights stay bf16, no AMP needed;
        # fp16 on Turing/Volta via TRUE AMP: backbone loads fp32 and autocast
        # runs matmuls in fp16. Pure-fp16 WEIGHTS overflow inside attention
        # softmax at long sequences (observed: every step non-finite on T4);
        # fp32 weights + fp16 compute is the canonical stable pattern.
        dtype = {"auto": torch.bfloat16 if cc >= 80
                 else (torch.float16 if cc >= 70 else torch.float32),
                 "bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[args.dtype]
        props = (f"| gpu={p.name} cc={p.major}.{p.minor} "
                 f"vram={p.total_memory / 2**30:.0f}GiB ")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        # CPU path: fp16 is unsupported (LayerNorm), fp32 doubles RAM vs bf16
        # and bf16 forward/loss verified working locally — so bf16 unless the
        # user explicitly asked for fp32.
        dtype = torch.bfloat16 if args.dtype != "fp32" else torch.float32
    print(f"[probe] grok-probe start | model={MODELS[args.model]} | device={device} "
          f"{props}| dtype={dtype} | sample_size={args.sample_size} "
          f"bs={args.batch_size} max_steps={args.max_steps} seed={args.seed}",
          flush=True)

    tok, model, llm_dim = load_backbone(args.model, args.limit_layers, device, dtype)
    if device == "cuda" and dtype == torch.float16:
        # TRUE AMP for fp16 GPUs: weights must be FP32 (pure-fp16 weights
        # overflow attention softmax at L~5000 — every step went non-finite).
        # autocast handles per-op precision; VRAM cost is 2x weight bytes
        # (~9 GiB backbone on T4) but activations stay half.
        print("[probe] fp16 GPU: reloading backbone in fp32 + autocast "
              "(true AMP — pure-fp16 weights overflow in softmax)", flush=True)
        model = model.to(torch.float32)
    if args.ram_cap_gib:
        # AFTER the loader: the safetensors mmap counts against RLIMIT_AS even
        # though it's file-backed (page-cache), so capping before load would
        # refuse to mmap the weights. From here on the cap bounds everything
        # else: activations, streamed row groups, projector state.
        enforce_ram_cap_gib(args.ram_cap_gib)
    # Projector stays FP32 when GradScaler is active: it refuses fp16 grads
    # ("Attempting to unscale FP16 gradients"), and fp32 master weights are
    # the canonical AMP pattern anyway (25M params ~ 0.1 GiB — negligible).
    # embeds_for casts its output into the model dtype.
    proj_dtype = torch.float32 if (device == "cuda"
                                   and dtype == torch.float16) else dtype
    proj = HourglassProjector(VISION_DIM, llm_dim).to(device=device,
                                                      dtype=proj_dtype)
    for p in proj.parameters():
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"[probe] HourglassProjector params = {n_params:,} | LN({VISION_DIM}) -> "
          f"Linear({VISION_DIM},{2 * llm_dim}) -> GELU -> Linear({2 * llm_dim},{llm_dim})",
          flush=True)

    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    # fp16 needs loss scaling (bf16/fp32 don't); without it grads underflow
    # and the loss goes non-finite at ~step 2 (observed on T4).
    # init_scale=1: default 65536 overflows fp16 grads on long sequences and
    # needs ~17 halvings to calibrate — our 5-skip FATAL aborts mid-ramp.
    scaler_enabled = device == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled,
                                  init_scale=1.0, growth_interval=10 ** 9) \
        if device == "cuda" else None
    monitor = ProbeMonitor()
    records: list[dict] = []
    consecutive_bad = 0

    # ---- resume: local first, then the HF bundle pushed by an earlier run ----
    start_step, opt_state = 0, None
    latest_sd = os.path.join(ckpts, "latest.safetensors")
    latest_opt = os.path.join(ckpts, "latest.opt.pt")
    if args.resume and not os.path.exists(latest_opt):
        _phase("no local checkpoint — pulling HF bundle from a previous run ...")
        if pull_bundle(args.out_dir):
            import shutil
            for fn in ("latest.safetensors", "latest.opt.pt"):
                src = os.path.join(args.cache_dir, fn)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(ckpts, fn))
            for fn, dst in ((LOG_FILE, log_path), (CURVES_PNG, curves_path)):
                src = os.path.join(args.cache_dir, fn)
                if os.path.exists(src):
                    shutil.copy(src, dst)
            print("[probe] checkpoint bundle restored from HF", flush=True)
    skip_rows = 0
    if args.resume and os.path.exists(latest_opt):
        st = torch.load(latest_opt, map_location=device, weights_only=False)
        from safetensors.torch import load_file
        proj.load_state_dict({k: v.to(device) for k, v in load_file(latest_sd).items()})
        opt_state = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in st["opt"].items()}
        start_step = st["step"]
        skip_rows = start_step * args.batch_size   # deterministic plan => exact replay
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get("type") == "train":
                            records.append(r)
                    except json.JSONDecodeError:
                        pass
        if records:
            monitor.ema = records[-1]["ema_loss"]
        print(f"[probe] RESUME from completed step {start_step} "
              f"(fast-forwarding {skip_rows} streamed rows)", flush=True)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    _phase("dataset: manifest + remote key index (~15 min of HTTP). NO bulk "
           "download — row groups stream over HTTPS during training ...")
    manifest = fetch_manifest(args.cache_dir)
    stream_order = list_shards()
    # old Modal smoke shards have ONE 1360-row group (~9 GiB): unstreamable
    EXCLUDED_SHARDS = {"data/emb_0000.parquet", "data/emb_0001.parquet"}
    stream_order = [s for s in stream_order if s not in EXCLUDED_SHARDS]
    random.Random(args.seed).shuffle(stream_order)   # same shuffle as the plan
    index = build_key_index(stream_order, cache_dir=args.cache_dir,
                            rebuild=args.rebuild_index)
    plan = build_epoch_plan(manifest, index, args.sample_size, args.seed,
                            excluded_shards=EXCLUDED_SHARDS)
    n_planned = sum(len(v) for v in plan.values())
    del index                                   # ~100 MB of strings, not needed
    _phase(f"dataset ready: SUBSAMPLED {n_planned} of {len(manifest)} manifest "
           f"rows (sample_size={args.sample_size}, seed={args.seed}) across "
           f"{len(plan)} shards; delivery shard-major (streaming concession) "
           f"— normalize grok windows per samples_seen.")

    collate = make_collate(tok, tok.pad_token_id, args.max_seq_len)
    steps_per_epoch = n_planned // args.batch_size
    steps_total = min(max(args.max_steps, 1), args.epochs * steps_per_epoch)
    print(f"[probe] steps/epoch={steps_per_epoch} @bs{args.batch_size} | planned "
          f"steps={steps_total} | grok reference ~{SAMPLES_PER_BASETEN_GROK} "
          f"samples_seen (~step {SAMPLES_PER_BASETEN_GROK // args.batch_size})", flush=True)
    for g in opt.param_groups:
        g["lr"] = lr_at(start_step + 1, steps_total, args.lr, args.warmup)

    def _batch_iter():
        """Infinite batch generator over the streaming dataset with epoch wrap;
        rows already consumed by resumed steps are skipped exactly (skip only
        applies on the first pass — later epochs replay the full plan)."""
        first = True
        while True:
            ds = EmbStreamDataset(plan, stream_order,
                                  start_pos=skip_rows if first else 0,
                                  rg_cache_dir=os.path.join(args.cache_dir,
                                                            "rg_cache"))
            first = False
            loader = torch.utils.data.DataLoader(
                ds, batch_size=args.batch_size, drop_last=True,
                collate_fn=collate, num_workers=args.workers,
                persistent_workers=args.workers > 0,
                pin_memory=(device == "cuda"))
            yield from loader

    it = _batch_iter()
    logger = open(log_path, "a" if start_step else "w", buffering=1)
    # Provenance header — first JSONL record is the verbatim config + git SHA
    # + args (manifest hash not yet known — a richer run_start follows after
    # fetch_manifest). Runs are comparable from the log alone (Karpathy/Huyen).
    if not start_step:
        try:
            hdr = _config_header(_CFG, manifest_path=None,
                                 extra={"run": "grok_probe_qwen", "args": vars(args)})
            logger.write(json.dumps(hdr) + "\n")
        except Exception as e:
            print(f"[probe] WARNING: could not write config header ({e})", flush=True)
    t0 = time.time()
    step = start_step
    samples_seen = start_step * args.batch_size
    tokens_total = 0

    while step < steps_total:
        try:
            batch = next(it)
        except StopIteration:                   # dataset shorter than planned
            break
        step += 1
        out = train_step(model, proj, opt, batch, device, scaler=scaler)
        if not out["finite"]:
            consecutive_bad += 1
            # GradScaler skips the optimizer step on inf/NaN grads — that's a
            # normal recovery path in fp16. Only abort if it persists.
            print(f"[probe][WARN] skipped step {step} (loss={out['loss']:.4f} "
                  f"gnorm={out['gnorm']:.3g} scale={scaler.get_scale():.3g}; "
                  f"{consecutive_bad} consecutive)", flush=True)
            if consecutive_bad >= 5:
                print(f"[probe][FATAL] 5 consecutive bad steps at {step} — "
                      f"training is diverging; last good state: {latest_sd}",
                      flush=True)
                logger.close()
                raise SystemExit(1)
            continue
        consecutive_bad = 0
        samples_seen += out["batch_size"]
        tokens_total += out["tokens"]
        elapsed = max(1e-9, time.time() - t0)
        rec = {
            "type": "train",
            "step": step,
            "epoch": step // max(1, steps_per_epoch),
            "loss": round(out["loss"], 5),
            "gnorm": round(out["gnorm"], 4),
            "lr": float(opt.param_groups[0]["lr"]),
            "tokens": out["tokens"],
            "samples_seen": samples_seen,
            "ts": round(time.time(), 1),
        }
        monitor.update(step, rec["loss"], samples_seen)
        rec["ema_loss"] = round(monitor.ema, 5)
        rec["tok_s"] = round(tokens_total / elapsed, 1)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["it_s"] = round((step - start_step) / elapsed, 3)
        rec["step_ms"] = out["step_ms"]
        records.append(rec)
        logger.write(json.dumps(rec) + "\n")

        if step % STATUS_EVERY == 0 or step == steps_total:
            eta_min = (steps_total - step) * rec["step_ms"] / 1000 / 60
            peak = (f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB "
                    if device == "cuda" else "")
            print(f"[probe] step {step}/{steps_total} loss={rec['loss']:.4f} "
                  f"ema={rec['ema_loss']:.4f} gnorm={rec['gnorm']:.2f} "
                  f"lr={rec['lr']:.2e} tok/s={rec['tok_s']:.0f} "
                  f"samples_seen={samples_seen} {peak}{rec['it_s']:.2f}it/s "
                  f"ETA={eta_min:.0f}m"
                  + (f" ALERTS={monitor.n_alerts}" if monitor.n_alerts else ""),
                  flush=True)
        if step % CHART_EVERY == 0 or step == steps_total:
            render_curves(records, curves_path)
        if step % SAVE_EVERY == 0 or step == steps_total:
            save_projector(proj, opt, step, latest_sd)
            push_bundle(args.out_dir, ckpts, step)   # best-effort; needs HF_TOKEN

    save_projector(proj, None, step, os.path.join(args.out_dir, FINAL_PATH))
    render_curves(records, curves_path)
    final_loss = records[-1]["loss"] if records else float("nan")
    logger.write(json.dumps({
        "type": "run_end", "step": step, "samples_seen": samples_seen,
        "final_loss": final_loss, "final_ema": monitor.ema,
        "collapse_step": monitor.collapse_step,
        "collapse_samples_seen": (monitor.collapse_step or 0) * args.batch_size,
        "n_alerts": monitor.n_alerts, "n_banners": monitor.n_banners,
        "wall_min": round((time.time() - t0) / 60, 1)}) + "\n")
    logger.close()
    push_bundle(args.out_dir, ckpts, step)
    print(f"[probe] DONE step={step} samples_seen={samples_seen} "
          f"final_loss={final_loss} ema={monitor.ema} "
          f"collapse_at_samples_seen={monitor.collapse_step and monitor.collapse_step * args.batch_size}",
          flush=True)
    print(f"[probe] artifacts: {log_path} | {curves_path} | "
          f"{os.path.join(args.out_dir, FINAL_PATH)}", flush=True)
    print("[probe] send back: probe_log.jsonl + probe_curves.png + final loss "
          "+ samples_seen at any observed collapse", flush=True)


if __name__ == "__main__":
    main()
