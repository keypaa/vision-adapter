# TRAINING_PLAN — 7-phase roadmap for the Vision-Adapter trainer + HF publish

Authoritative record of what we decided and why. Each phase has a **Report**
section to be filled in as the phase runs, so later phases (and future us) can
see the decisions and numbers that shaped the work.

Status legend for Report sections:
- `[ ] TODO` — not started
- `[~] IN PROGRESS`
- `[x] DONE`

---

## Context / facts established before Phase 1 (verified, not assumed)

| Topic | Verified fact |
|---|---|
| DeepSeek-V4 attention | **Eager-only.** `modeling_deepseek_v4.py` sets `_supports_flash_attn=False`, `_supports_sdpa=False`, `_supports_flex_attn=False` because `head_dim=512` (FA2/3/4 cap at 256; SDPA lacks the per-head sink term; FlexAttention breaks on the compressor KV-append). ⇒ **No flash-attention version works on the LLM, on any GPU.** Do NOT set `attn_implementation` — it would be silently ignored. |
| FA2.7.4 wheel | Used only by the MoonViT **precompute**, which is **done**. Not part of training. |
| Trainer execution state | **Never run.** No `checkpoints/`, `logs/`, or `dryrun_report.txt` on the Volume. |
| Embedding HF repo (`vision-adapter-embeddings`) | **Empty** — the raw-`.pt`-blob `push_embeddings_to_hf` committed nothing before being stopped. |
| Manifest HF repo (`vision-adapter-manifests`) | **Fully pushed** (train / val / cauldron). |
| Image HF repo (`vision-adapter-images`) | **8 / 17 shards** present; 9 remaining; `push_image_corpus_to_hf` is resumable per shard. |
| Corpus / volume | ~139k embeddings (~950 GB `.pt`), 79,659 agentic + ~59,328 cauldron images. Training manifest = 120k subset (54k agentic / 54k doc / 12k conv) + ~2.4k val. |
| B300 GPU | 288 GB VRAM / 8 TB/s, ~$7.10/h on Modal, SM103 Blackwell. Requires torch ≥ 2.7 + cu128 (current pin `torch==2.5.1` cannot run it). Quantized DeepSeek ~155-167 GiB fits in VRAM, so CPU-offload/`device_map` can be dropped. |
| Step math | `BATCH_SIZE=8`, `MAX_SEQ_LEN=4096`, `EPOCHS=2`, 120k train ⇒ **30,000 steps**. Grok-window telemetry is samples-based (`samples_seen` ~58k ⇒ step ~7,200-11,000 at bs 8). |
| I/O bottleneck | **Unmeasured.** `it_s ~0.1` (TELEMETRY) suggests backward-through-304B is the wall, but the I/O share is unknown — that is exactly what Phase 1 measures before any rewrite. |

---

## Phase 1 — Data-path micro-benchmark

**Goal:** measure the real cost of loading `.pt` embeddings from the Modal
Volume before deciding how much of the data path to rework.

**Details:**
- Add a small Modal probe (in `modal_pipeline.py`, `_etl_image`, no GPU).
- Sample ~200 `.pt` files from `/data/embeddings/`, time `torch.load(..., weights_only=True)` per file.
- Report: avg / p95 latency per file, MB/s, and projected **I/O share of a
  training step** at bs=8, num_workers=2 vs 8.
- Output must follow the AGENTS.md logging contract (flush=True, phases, heartbeats).

**Decision rule (agreed):**
- I/O share < ~15% of a ~10 s A100 step → parquet rework is about HF-publish
  quality (still worth it), not training speed.
- I/O share > ~40% → the parquet rework is also the training-speed fix.

### Report — Phase 1
- `[x]` Status: DONE (2026-08-18, `emb_io_bench`, 200 sampled .pt, avg 7.9 MB/file)
- Per-file latency (avg / p95):
  - **Cold** (fresh container, first touch): avg **463 ms**, p95 **873 ms**, **17.2 MB/s** serial
  - **Warm** (Modal Volume block cache, immediate re-run): avg **7 ms**, p95 **27 ms**, **1153 MB/s** serial
- MB/s achieved: 17 MB/s cold serial → **2006 MB/s @ 8 workers (cold, latency hidden)** → 3847 MB/s @ 2 workers warm
- Projected I/O share per step (workers=2 / workers=8):
  - Cold serial (worst case): ~3.7 s/batch → ~37% of a 10 s step
  - Cold @ 8 workers: ~0.46 s/batch → ~5%
  - Warm (any workers): ~1% or less — **negligible**
- Verdict: **publish-quality, not training-speed.** The trainer's DataLoader workers already overlap cold latency, and once the Volume's block cache warms (epoch 0), per-file `torch.load` is ~7 ms and I/O is ~1% of a step. The parquet rework stays worthwhile for HF-publish quality + volume recovery, NOT to fix a training I/O bottleneck.
- Notes: the two-run spread (463 ms → 7 ms) is the Volume's on-container block cache — repeat reads are fast, cold reads are brutal. If the trainer ever runs on a fresh container each step it would starve, but persistent_workers + an epoch-0 warm pass eliminates it. Consider keeping the phase-3 DataLoader bump (workers=8) as cheap insurance; full ParquetEmbSFT is optional for speed but still valuable for HF.

---

## Phase 2 — `pack_embeddings_to_parquet` (single source of truth)

**Goal:** convert the 139k `.pt` embeddings into ~100 large parquet shards on
the Volume (`/data/shards/`), so one format serves both the trainer and the HF
publish.

**Details:**
- New resumable fn in `modal_pipeline.py` (Volume write + `vol.commit()` per shard, skip existing shards).
- **~1360 rows/shard ⇒ ~100 shards, ~10 GB each** (agreed size).
- Columns: `key` (`embeddings/<sha1>.pt`, byte-identical to the manifest `emb`
  field ⇒ **100% compatible with existing manifests**), `n_vis` (int),
  `vis_bytes` (raw BF16 `tobytes()` — no torch.save pickle, no compression).
- 32-way parallel Volume reads, per-shard write + commit, resumable.
- Writes `dataset_info.json` + dataset card in the shard dir.
- **Verification gate:** shard row-sum == glob count; spot-check
  `frombuffer → reshape(-1, 4096)` equals `torch.load` of the original `.pt`.
- Covers both the 120k training-referenced embeddings AND the full corpus, so
  the HF publish is complete and the volume is recoverable.

### Report — Phase 2
- `[~]` Status: IN PROGRESS — smoke DONE on Modal (2 shards, `emb_0000/0001`, ~18 GB, round-trip verified); full corpus moved to **`local_pack.py --hf-only`** (Phase 7 implementation), running on the owner's laptop.
- Shard count / row-sum vs glob count: 138,987 rows ⇒ **103 shards** @1360 (last = 267).
- Total GB on volume (`/data/shards`): 18 GB (2 smoke shards only; full corpus goes to HF, not the volume — see Phase 6.2 update).
- Round-trip equality check: **pass** (`frombuffer→view(bf16)→reshape(-1,4096)` pinned by tests).
- Pack wall-time: Modal path volume-latency-bound (~508 s/shard cold) ⇒ superseded by local packer (network-bound, ~3.5 min/shard projected).
- Notes: writer uses `compression=None` (bf16 incompressible; measured 2.3× faster writes) + pipelined next-shard download against HF push.

---

## Phase 3 — Trainer reads shards (`ParquetEmbSFT` replaces `EmbSFT`)

> **SKIPPED BY MEASUREMENT (2026-08-21).** Re-bench of `.pt` reads: warm **16 ms/file** serial (499 MB/s), **1727 MB/s @ 8 workers**, I/O ≈ **0–1 % of a step** with `num_workers=8` (now applied). The parquet rework bought nothing for training speed; trainer stays on `EmbSFT` reading `/data/embeddings/*.pt`. Parquet is the *publish/reproduction* format only.

**Goal:** remove per-file `torch.load` from the training hot loop.

**Details:**
- `modal_train.py`: replace `EmbSFT` with `ParquetEmbSFT(torch.utils.data.Dataset)`.
- Same returned row contract (`vis, user, assistant, g`) so
  `make_collate` / `inject_visual` / `_one_step` are **untouched**.
- Global index → `(shard, row)`; per-worker LRU of loaded shard tables
  (~1-2 shards ≈ 10-20 GB/worker, inside the 200 GB cap).
- Reconstruct tensor: `np.frombuffer(uint8) → torch.bfloat16 → view(-1, 4096)`.
- DataLoader: `num_workers=8` (was 2), `pin_memory=True`.

### Report — Phase 3
- `[ ]` Status: TODO
- `n_vis` / shape validation across shards:
- Per-step I/O time before vs after (vs Phase 1 baseline):
- Notes:

---

## Phase 4 — A100 `train_dryrun` first

**Goal:** validate the **never-run training code** in isolation on known
hardware + the known-good image (`torch==2.5.1`), before any GPU-stack churn.
NOT a speed contest — but it records the step-time baseline that Phase 5's
B300 dryrun is compared against (added to `train_dryrun` 2026-08-21: 4 timed
steps after warmup, printed + written to `dryrun_report.txt`).

**Details:**
- Run `modal run modal_train.py::train_dryrun` on **A100-80GB**, dataset = current **`EmbSFT`** (.pt path — see Phase 3 skip).
- Keep `BATCH_SIZE=8`, `MAX_SEQ_LEN=4096`, `GPU_MEM_CAP_GIB=70`, grad checkpointing ON.
- Gate: memory-box PASS **+ recorded `step=Xs (Y it/s)` baseline** in `dryrun_report.txt`.

### Report — Phase 4
- `[ ]` Status: TODO
- Dry-run verdict (PASS/FAIL, peak GiB):
- Step time baseline (it/s) on A100 with `.pt` path:
- Notes:

---

## Phase 5 — B300 stack (only after Phase 4 passes)

**Goal:** move training to B300 with the correct (and minimal) stack changes.

**Details:**
- `train_image`: `torch==2.5.1` → **torch ≥ 2.7 (cu128)**; keep transformers /
  accelerate / datasets recent; ensure `pyarrow` present.
- Drop `device_map="auto"` CPU-offload on B300 (quantized model ~155-167 GiB
  fits in 288 GB).
- **Keep gradient checkpointing ON until the B300 dryrun passes** — do not
  trust the "54 GB activations / 67 GB free" estimate; verify with a real
  dryrun before disabling.
- Do NOT touch `attn_implementation` (eager-only model).
- Gate: B300 `train_dryrun` PASS under a new memory gate (e.g. ~250 GiB),
  plus recorded step time.

### Report — Phase 5
- `[ ]` Status: TODO
- B300 dry-run verdict (PASS/FAIL, peak GiB):
- Step time (it/s) on B300:
- Grad checkpointing ON/OFF decision + measured trade-off:
- Notes:

---

## Phase 6 — HF publishes (finish the money story)

**Goal:** publish the full data to HF for (a) public access and (b) volume
recovery, and stop the un-finished pushes.

**Details:**
1. **Finish images:** re-run `push_image_corpus_to_hf` to push the remaining
   **9 shards** (resumable; per-shard read/write/upload timing lines already
   printed). Dataset card is pushed at the END (`README.md` + `dataset_info.json`),
   guarded by `if ... not in remote`.
2. **Embeddings via parquet:** new `push_embeddings_shards_to_hf` uploads
   `/data/shards/*.parquet` to `keypa/vision-adapter-embeddings`, one commit
   per shard, skipping present ones. This **replaces** the raw-`.pt`-blob
   `push_embeddings_to_hf` (HF repo is empty ⇒ no back-compat issue).
   - Parquet is the right HF format: native viewer, streaming, range reads,
     and a complete copy of the volume for recovery.
3. Do **not** delete `/data/embeddings/*.pt` until a full training run finishes
   AND the HF parquet copy is verified (the Volume is the trainer's read source).

### Report — Phase 6
- `[x]` 6.1 Images: **DONE — 17/17 shards** (138,987 rows) live on `keypa/vision-adapter-images`.
- `[~]` 6.2 Embeddings: **superseded by `local_pack.py --hf-only`** (runs on the owner's laptop, pushes `data/emb_XXXX.parquet` straight to HF; target **103/103**). Volume copy `/data/shards` no longer part of the plan (trainer reads `.pt`; hydrate-from-HF possible later if ever needed).
- HF repo sizes (images / embeddings): TBD after tonight's pack.
- Verification: `load_dataset` round-trip OK: pending post-pack spot-check.
- Notes: dataset cards for all three repos + a linking HF space = future polish.

---

## Phase 7 — Local push variant

> **DONE (implemented as `local_pack.py`, 2026-08-21).** Standalone resumable CLI:
> streams each shard's `.pt` from the volume, packs the identical
> `key`/`n_vis`/`vis_bytes` schema (`compression=None`, 64-row batches),
> pushes to HF; `--hf-only` skips the volume copy; `--only i[:j]` ranges;
> pipelined download↔push; progress JSONL+PNG. Byte-compatibility pinned by tests.

### Report — Phase 7
- `[~]` Status: code DONE + smoke pending on home fiber (shard 2 first, then full rip).
- Upload speed achieved (MB/s, vs 1 Gb/s fiber): TBD tonight.
- Shards pushed locally (target 103/103): TBD.
- Byte-equality vs Modal-packed shards: schema/order identical by construction (same sorted enumeration); round-trip equality covered by test suite.
- Notes:

---

## Cross-cutting decisions (locked)

- **Parquet shards = single source of truth** for both training reads and the
  HF publish. `key` column matches the manifest `emb` field exactly ⇒ no
  manifest change, 100% compatibility with existing manifests.
- **Shard size:** large ~10 GB shards (~1360 rows) — agreed.
- **First training validation:** A100 dryrun with the new data path, THEN the
  B300 stack. Isolates data-path vs GPU-stack variables.
- **No flash-attention work for the LLM** — eager-only model, head_dim=512,
  nothing to enable on any GPU.
- **B300 image upgrade is torch/cu128 + offload removal**, not attention work.
- **Keep `.pt` files on the Volume until training completes + HF verified.**
