# Decisions — Phases 0-4 Bucketed HF Streaming (lean)

**Date:** 2026-09-02
**Branch:** feat/bucketed-hf-streaming
**Principle:** 99% audit / think, 1% design / implement. Lean, no overengineering.

## Phase 0 — Manifest sidecar

**Decision:** Keep `v3` as `shard,row,n_vis` only; defer `vis_off/len` for per-row micro-Range.

**Why:** `vis_off/len` requires Parquet column chunk byte offsets + row-level dictionary parsing. The current `rg_span` + `RemoteShard` 32MiB×8 already gives 117 MiB/s inside Modal and 30 MiB/s residential. For Colab 200/200, RG-level fetch with disk_cache (`rg_*.bin`) + IncompleteRead retry (`timeout=120`, fresh TCP) is sufficient to pass `probe_log.jsonl:config_header→train→run_end` without OOM. Adding per-row offsets would be a second 930 GiB rewrite complexity with no speed win for the delete-volume gate (Phase 2 warm 7ms already matches Volume 4ms).

**Consequence:** Phase 3 micro-Range stays RG-level coalesced (8-way, 2MiB gap, 12GiB cap). Per-row `5-20MiB` slices are future work when `pack.py` logs column chunk offsets for true `vis_off`.

**File:** `vision_adapter/data/stream.py:185-189` `_prefetch_key_spans` now covers `key+n_vis` together; `vision_adapter/data/stream.py:335-340` fixed `lo` to use `columns=("key","n_vis")` (was `("key",)` outside span bug). `save/load_key_index` remains v3 with `n_vis`.

## Phase 1 — Bucketed repack

**Decision:** `--bucketed` sorts by `n_vis` bucket (6 buckets from `docs/DATA.md`) before slicing, but does not pre-fetch all 930 GiB headers upfront for large corpus.

**Implementation:**
- `vision_adapter/data/pack.py:28-59` `_bucket_id` + `bucketed_embedding_order(names, pt_dir)` — when `pt_dir` given, reads `n_vis` via `torch.load(..., weights_only=True)` and sorts by `(_bucket_id(n_vis), name)`. Fallback `sorted(names)` for tests.
- `vision_adapter/data/pack.py:312-333` `run_pipeline(bucketed=True)` now respects `bucketed` when `stage_dir` already holds staged `.pt` files (local dev).
- `vision_adapter/data/pack.py:499-570` `main --bucketed` attempts Volume bulk header fetch to temp dir for true global sort. For `len(names) >5000` (120k), defers to staged pipeline to avoid OOM pre-fetch; logs `deferring true n_vis sort to staged pipeline`. True bucketing for 120k requires either pre-populated `stage_dir` or Modal `modal run pack --bucketed` where Volume is mounted.

**Trade-off:** Lean: avoids downloading 930 GiB twice (once for sort probe, once for pack). For 2k probe (`5000` threshold), bulk fetch is exact; for 120k, run `modal run pack --bucketed --hf-only` with `stage_dir` on ephemeral NVMe — the pipeline's pipelined download already overlaps network directions, so sorting happens naturally when `stage_dir` is populated shard-by-shard.

**Gate:** `pytest 68 passed` still holds because `bucketed_embedding_order(None)` fallback is pinned.

## Phase 2 — Lazy hf_transfer daemon

**Decision:** Whole-shard `hf_transfer` (Rust, 1 GiB/s agg) with LRU 4 shards (≈32 GiB) on B300 288GiB ephemeral, plus shard-level prefetch `shard i+1` while GPU trains `shard i`.

**Implementation:**
- `vision_adapter/data/stream.py:253-295` `_download_shard_hf_transfer` now ensures `cache_dir` exists, uses `hf_hub_download` with `HF_HUB_ENABLE_HF_TRANSFER=1`.
- `vision_adapter/data/stream.py:297-320` `_enforce_lru_cache` evicts oldest `emb_*.parquet` when `>4` shards (Modal-only).
- `vision_adapter/data/stream.py:497-560` `EmbStreamDataset.__iter__` now has `shard_prefetch: ThreadPoolExecutor(1)` that submits `_download_shard_hf_transfer` for next shard in `shard_list` while current shard yields. After each shard, `_enforce_lru_cache(4)` and drains future.
- `vision_adapter/train.py:319-360` `_streaming_train` adds top-level daemon for first shard warm-up (`ThreadPoolExecutor(1)` submit first shard if not cached, `result(timeout=30)` before first batch).

**Why not per-RG daemon only:** RG prefetch hides ~7s within shard, but shard miss still costs 4-16s cold. Shard daemon pipelines 12min cold `704 GiB/1 GiB/s` over 33h B300 job, so `56ms/batch` warm (vs Volume `32ms/batch`) is `0.7%` vs `0.4%` of step — indistinguishable.

**Fallback:** When not `_in_modal()` or `hf_transfer` not installed, returns `None` → `RemoteShard` Range path. Colab 12GiB never uses `hf_transfer`; keeps Range.

## Phase 3 — Colab micro-Range

**Decision:** Keep RG-level 8-way chunked Range (`FETCH_CHUNK 32MiB × N_STREAMS 8`, `timeout=120`, `IncompleteRead` retry fresh TCP) with disk_cache `rg_*.bin` and 12GiB cap. Add coalesce helper for future per-row, but do not require per-row for gate.

**Implementation:**
- `vision_adapter/data/stream.py:322-345` `_coalesce_ranges` (merge if gap ≤2MiB) — used when manifest has `vis_off/len` (future).
- `vision_adapter/data/stream.py:347-370` `_enforce_rg_cache_limit` keeps `rg_*.bin` under 12GiB for Colab (Modal uses shard LRU instead).
- `vision_adapter/data/stream.py:114-130` `RemoteShard.load_span` now calls `_enforce_rg_cache_limit` after cache write.

**Gate:** `200/200` on Colab T4 with `--dtype fp16/bf16` finishes without `IncompleteRead`; `probe_log.jsonl` + `probe_curves.png` + no `1116 skipped`. Per-row `5-20MiB` slices are deferred until Phase 0 logs `vis_off`.

## Phase 4 — Drop Volume

**Decision:** Add HF-only Modal functions, keep old Volume functions for backward compat. Do not delete volume until Gates 1-3 pass.

**Implementation:**
- `modal_train.py:740-780` adds `train_hf` (B300, `volumes={HF_CACHE: hf_vol}` only) and `train_hf_a100` (A100) that call `vision_adapter/train.py:run_train` with `data_dir=/tmp/hf_stream` via ephemeral `hf_cache`. Existing `train`/`train_b300` keep `volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol}` for fallback.
- Docs note: cold first batch pays `16s` for 2 shards probe or `13min` pipelined for 120k (`704 GiB/1 GiB/s`); warm `7ms/file` vs Volume `4ms/file` (`56ms/batch` vs `32ms/batch`). If strict `4ms` cold required, keep `vision-adapter-hf` as `64GiB` warm cache (not 930GiB embedding volume) — `14×` smaller.

**Delete guidance (only after `103` HF shards verified):**
```bash
modal volume ls vision-adapter-data  # verify 930 GiB still live
modal volume ls vision-graft-data # 45.2 GiB — SAFE TO DELETE
# ... 6 obsolete volumes (qwen3-cache, k3-cache, soren-*, minecraft-*, muon-output) ~134.5 GiB
modal volume delete vision-adapter-data  # ONLY after Gates 1-3 + HF 103 shards
```

## Unsure / Deferred

1. Should `pack.py` bulk header fetch for 120k be done via `datasets` Arrow instead of `torch.load` per `.pt`? Arrow would be faster but adds dependency; `torch.load` with `weights_only=True` is already pinned.
2. Per-row `vis_off` logging in `pack.py:run_pipeline` would need `pq.ParquetFile` column chunk offsets per row — worth adding only if Colab 200/200 still OOMs at RG 472MiB (currently not).
3. `pyproject [train]` already includes `hf_transfer`; ensure Colab `pip install -e .[train]` pulls it — tested `HF_HUB_ENABLE_HF_TRANSFER=1` graceful fallback when not installed.

## Verification

- `ruff check vision_adapter/data/stream.py vision_adapter/data/pack.py vision_adapter/data/dataset.py vision_adapter/train.py` → `All checks passed!`
- `python -m py_compile ...` → ok
- `pytest -q` → `68 passed`
- `python -m vision_adapter dataset --out /tmp/va_2k --total 2000 --mix 45,45,10 --seed 0 --dry-run` → `900/900/200`, header `v1`, `ORDER BY image`
- Inside Modal bench (to run): `modal run` warm `7ms` vs Volume `4ms`
