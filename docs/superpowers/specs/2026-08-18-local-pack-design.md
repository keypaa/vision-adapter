# Local Embedding Pack (`local_pack.py`) — Design

**Date:** 2026-08-18
**Status:** Approved (design phase; implementation plan follows)

## Context

The Modal-side pack (`pack_embeddings_to_parquet`) is **volume-latency-bound, not
compute-bound**: cold volume reads serve ~3-4 files/s, the block cache holds far
less than the 950 GB corpus, so a full Modal pack runs **~11h+ regardless of
workers** (smoke test: shard 0 load 18s warm, shard 1 load 508s cold).

The pack itself is trivially light: per row it is a `torch.load`, a bf16
`tobytes()`, and a parquet binary write. This work can move entirely onto the
user's laptop (6 cores / 6 threads, 16 GB RAM, 1 Gb/s fiber) and run overnight,
streaming the data in ~10 GB shard-sized chunks — no Modal compute, no GPU, low
CPU temperature.

**Decided destination (per user):** packed shards go to **both** the Modal
Volume `/data/shards/` (trainer source of truth) **and** HF
`keypa/vision-adapter-embeddings` (publish + comparison in Phase 3).

## Goal

Replace the Modal pack path with a local, resumable, RAM-bounded, cool-running
script that produces **byte-compatible parquet shards** (`emb_{i:04d}.parquet`,
schema `key`/`n_vis`/`vis_bytes`) covering the full corpus (~139k embeddings,
~100 shards), uploads each to the Volume, and pushes each to HF.

## Constraints

- 16 GB RAM hard cap → never materialize a whole shard's rows in memory.
- 6 cores / 6 threads → download pool capped at 6; `torch.load` sequential; no
  multiprocessing. Network-bound, so cores stay mostly idle.
- Resumable/idempotent: a shard pushed to HF is never re-pushed; a half-finished
  shard resumes cleanly after interruption.
- Byte compatibility: identical shard boundaries (sorted file order, 1360
  rows/shard) and identical row bytes to the Modal pack, so `emb_0000/0001`
  already on the Volume stay valid and manifests stay compatible.

## Data flow

```
vol.listdir("embeddings") → sorted names (same order as Modal's sorted(glob))
  shard i = names[i*1360 : (i+1)*1360]

  ┌─ already on volume AND on HF → skip
  ├─ on volume, not on HF       → pull parquet from volume → push to HF (no re-pack)
  └─ not on volume              →
       1. STAGE  download the slice via vol.read_file_into_fileobj (6 threads, retries)
       2. PACK   streaming pq.ParquetWriter; write table every 64 rows; free memory
       3. UPLOAD vol.batch_upload().put_file → /data/shards/emb_XXXX.parquet
       4. PUSH   huggingface_hub.upload_file → keypa/vision-adapter-embeddings data/emb_XXXX.parquet
       5. CLEAN  rm staging dir → next shard
```

## Components

### 1. Shard enumeration
- `vol.listdir("embeddings")` → list of `FileEntry` objects; take
  `entry.path` (`embeddings/<sha1>.pt`) and sort lexicographically (matches
  Modal `sorted(glob.glob(EMB_DIR/*.pt))` since all share the `embeddings/`
  prefix).
- Slice by `shard_rows = 1360`, same as the Modal pack.
- Shard filename: `emb_{i:04d}.parquet` (4-digit zero-padded).

### 2. STAGE (download)
- For each name in the slice, stream to local staging dir (default
  `/tmp/emb_stage/`) via `Volume.read_file_into_fileobj(path, fileobj)`.
- `ThreadPoolExecutor(max_workers=6)`; per-file retry 3× with backoff.
- A single shard ≈ 10 GB staged, then deleted after push.

### 3. PACK (streaming, RAM-bounded)
- Process staged files in batches of **64 rows**:
  - `torch.load(path, map_location="cpu", weights_only=True)`
  - validate `d.dim() == 2 and d.shape[-1] == 4096`
  - row = `{"key": "embeddings/<name>", "n_vis": int(d.shape[0]),
           "vis_bytes": d.view(torch.uint8).numpy().tobytes()}`
  - build a 64-row `pa.Table` and `ParquetWriter.write_table(...)`, then drop
    references → peak memory ≈ 64 × ~7 MB + writer buffers ≈ 0.5 GB.
- Write locally to `emb_XXXX.part.parquet`; rename to `emb_XXXX.parquet` only
  after the shard is fully written, so a crash leaves nothing half-uploaded.

### 4. UPLOAD to Volume
- `vol.batch_upload()` → `batch.put_file(local_path, "/data/shards/emb_XXXX.parquet")`.
- `vol.commit()` after upload.

### 5. PUSH to HF
- `huggingface_hub.upload_file(path_in_repo="data/emb_XXXX.parquet",
  path_or_fileobj=local_path, repo_id="keypa/vision-adapter-embeddings")`.
- HF upload uses resumable multi-part upload (1.25+ SDK).

### 6. Resume checks
- Volume check: `vol.listdir("shards")` → existing `emb_*.parquet`
  (`FileEntry.path`).
- HF check: `HfApi.list_repo_files("keypa/vision-adapter-embeddings")` → existing
  `data/emb_*.parquet`.
- Ordering per shard: if both → skip; if volume-only → pull+push to HF; else full
  pipeline. The 2 smoke shards (`emb_0000/0001` on Volume) are handled by the
  volume-only branch on the first run.

## Verification gate

Per shard:
- row count == slice length (1360, except possibly the final shard).
- Spot-check: `np.frombuffer(vis_bytes, np.uint16).view(torch.bfloat16) →
  reshape(-1, 4096)` equals `torch.load` of the original `.pt`.

Whole-run:
- `sum(n_vis)` and row-sum == `len(vol.listdir("embeddings"))`.
- Log heartbeats per shard: `[local-pack] shard i/N ... done/total, ETA, GB,
  stage/pack/upload/push seconds`.

## Config / CLI

Standalone script (`python local_pack.py`), no Modal decorators:
- `--shard-rows` (default 1360, must stay 1360 for compatibility)
- `--stage-dir` (default `/tmp/emb_stage/`)
- `--repo-ns` (default `keypa`)
- `--only i..j` (optional shard-range filter for testing / resuming a range)
- HF token via `HF_TOKEN` env (huggingface_hub default).
- Modal auth via default local profile (already configured).

## Dependencies (local venv already has these)

`modal`, `torch`, `pyarrow`, `numpy`, `huggingface_hub` — all present in
`.venv` (torch 2.13.0+cu130 CPU build works fine for `torch.load`).

## Out of scope

- The full `pack_embeddings_to_parquet` Modal fn stays in `modal_pipeline.py`
  (unused once the local script takes over, but kept for reference/resume).
- No GPU, no `precompute`, no image pushing.
- Deleting `/data/embeddings/*.pt` after pack (still guarded by Phase 6.3 rule:
  only after a full training run).
