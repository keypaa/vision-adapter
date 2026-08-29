# Best Practices — Filtered to Vision-Adapter

Branch: `refactor/discipline`. One-time investment in HOW to work (safer > faster).
Every item maps to a concrete `file:line` in this repo or is dropped — no generic advice.

---

## 1. From Karpathy — "A Recipe for Training Neural Networks" (2019)

> Source: http://karpathy.github.io/2019/04/25/recipe/

### 1.1 Become one with the data (step 1 of the recipe)

**Principle:** Before training, visually inspect thousands of examples, assess distribution, bias, duplicates, corruption, label noise. Code to filter/sort and inspect outliers.

**Why for Vision-Adapter:** Our `n_vis` spread 35→4900 was discovered late via the bucketing ladder — it drives both precompute cost and VRAM blowup. The visual token count should have been the first histogram plotted, not a debug finding.

**Repo mapping:**
- `preprocess.py:13-16` constants `MAX_PATCHES=65536`, `MAX_SIDE=7168` — never histogrammed against the corpus.
- `vision_adapter/models/precompute.py` greedy `pack_patched_batches` (historical: formerly inline in the monolith) in input order — fragmentation 10–20% because we never bucketed by `n_vis` first.

**Cost if skipped:** Silent tail-event dominance; 4900-token screenshots inflate `L` in `make_collate` → eager OOM (~46 GiB at bs8) that `_make_chunked_eager` papers over.

### 1.2 Set up the end-to-end skeleton + dumb baselines (step 2)

**Principle:** Build a tiny trusted pipeline *first*: fix seed, disable augmentation, evaluate on full test set, check loss at init, overfit a single batch.

**Karpathy's "overfit one batch" discipline:** Take only 2 examples, increase capacity, confirm you can reach minimal loss with predictions matching labels. If not, debug before proceeding. Do not add unverified complexity.

**Repo mapping:**
- `test_grok_probe_smoke.py` is the closest thing we have — but it uses a synthetic stub model (`_tiny_qwen`, 4 layers, 2K vocab) not the real `modal_probe.py`/`modal_train.py` data plane. No test uses `EmbSFT` + `make_collate` + `embeds_for` end-to-end on real manifest rows.
- `modal_probe.py:dryrun` (L4, `loss=5.61 peak=12.32GiB`) is now the production-plane equivalent — but it only landed in `3cb0d6f`. Before that, there was no memory gate.

**Cost if skipped:** We jumped 0→304B with one duplicated probe. A 2-example overfit would have caught the LayerNorm dtype bug (`EmbSFT float32 → bf16 projector`) before it hit Modal.

### 1.3 Overfit, then regularize, then tune (steps 3–5)

**Principle:** Find a large enough architecture that *can* memorize training data. Don't be a hero — copy the simplest proven design. Keep LR constant early (Adam 3e-4), add signals one at a time.

**Repo mapping:**
- `modal_train.py:49 LR=5e-4` is constant (no scheduler), while `grok_probe_qwen.py:104` and `modal_probe.py:65` have `lr_at` warmup 100 → cosine to 10%. Recipe inconsistency — a gate passing on probes may not reproduce on production.

**Cost if skipped:** Warmup exists to absorb the early `grad_norm` bursts (28–205 observed on Colab smoke). Without it, production hits `SPIKE-ALERT` at step 1.

### 1.4 Grounded in the Baseten vision-adapter recipe — the only public source

> Source: https://www.baseten.co/blog/glm-52-with-vision/ (and `baseten/GLM-5.2-Vision-NVFP4` it links)

Baseten's published recipe is the closest public bearing for our frozen-backbone adapter. Key numbers and why they anchor our probe:

| Baseten GLM 5.2 Vision (ground truth) | Vision-Adapter (our build) | Bearing |
|---|---|---|
| Kimi **K2.6** MoonViT-3d, 27 layers, **1152-dim** | MoonViT-V2 from **Kimi-K3**, 1024 hidden ×4 → 4096 after merge | Close but different tower revision — `moonvit.py` traces provenance; expect slight distribution shift, not a bug |
| Projector `1152→4608→6144` = **49.5M** (`pre_norm→linear1→GELU→linear2`, PatchMerger MLP) | Hourglass `4096→8192→4096` = **67,129,344** (`LN→Linear→GELU→Linear`), `ARCHITECTURE.md:22-23` | Structural mismatch is **expected** — different vision dim and LLM hidden (our 4096↔4096 coincidence is Kimi-K3 + DeepSeek-specific). Log discipline keeps "67M projector" vs "304B total / 67B active" separate |
| GLM-5.2 744B tot / A40B active, MoE+MLA+DSA, 1M context, 8×B200 Blackwell-only, `Glm5vForConditionalGeneration` + `sglang_glm5v` | DeepSeek-V4-Flash-0731 / Qwen3.5-2B via `transformers.AutoModelForCausalLM` | Different backbone families — same training *mechanism* (frozen backbone + trainable projector), different architecture |
| SFT: **66k** pairs, 2 epochs, batch **64** at **5e-4**, sharp drop near **step 900/1035** | Prod 120k bs8 = 15k/epoch; probe 2k/200 gate + 20k validation; `SAMPLES_PER_BASETEN_GROK = 900×64 = 57.6k` | Samples-normalized, not step-normalized — `step ≈ 57600/bs` is portable and documented in `DATA.md:75-92` |
| SFT: short Q/A only (long captions degraded alignment); RL: projector-only, 0→0.8 reward after first trace | Phase 8 SFT-only; RL out-of-scope `ARCHITECTURE.md:127`; manifest filtering does not yet enforce short-Q/A-only | Keep RL separate (your sequencing: data → precompute → probe → RL) |

**What is NOT in the Baseten pages (and not in Inference Engineering):** No `presidio-blossom` / `robo-r2` named; no training GitHub repo — only `plugins/*` + `truss/*` inside `baseten/GLM-5.2-Vision-NVFP4`. *Inference Engineering* (hardware Ch 3, local inference §3.5, Ch 4 software stack) is a serving book — it does not contain the GLM 5.2 SFT recipe. The blog is the sole SFT source.

---

## 2. From Chip Huyen — Designing Machine Learning Systems / MLOps

> Sources: https://github.com/chiphuyen/dmls-book/blob/main/README.md — https://huyenchip.com/mlops/

Huyen's book is organized around **design decisions over code snippets**: reliable, scalable, maintainable systems that move cleanly from offline experimentation to production, with shared reusable infrastructure for models, features, and observability.

### 2.1 Data versioning as code

**Principle:** Treat datasets like code — version the manifest, shard checksums, and upstream commits so a rebuild is provably identical.

**Repo mapping:**
- `vision_adapter/data/dataset.py` + `vision_adapter/manifest.py` seeded `random/numpy/torch` (historical: monolith did `random.seed(0)` only) only — Python-version-dependent; `1081` `LIMIT 54000` without `ORDER BY` makes the 54k agentic selection nondeterministic even with the seed. No `manifest_version`, `git_sha`, `upstream commits`, or `shard-set hash` in `train_manifest.jsonl`.
- `local_pack.py:54-60` `SCHEMA` has no per-row/per-shard `sha256(vis_bytes)`.

**Cost if skipped:** Delete the Volume → you cannot prove you rebuilt the same 120k. Future you cannot bisect a data bug.

### 2.2 Automate manual, error-prone workflows

**Principle:** If a step is currently manual (run this, then that, then check), automate it and make it observable.

**Repo mapping:**
- `vision_adapter/data/dataset.py` → `vision_adapter/data/cauldron.py` → manifest (`vision_adapter/manifest.py` header-first) → `vision_adapter/models/precompute.py` → `vision_adapter/data/pack.py` replaces the former 1800-line monolith (historical — see PIPELINE.md) with no per-file argument story and no local path without Modal.

**Cost if skipped:** Rebuild takes hours with no way to know where time goes — the specialized auditor had to reconstruct timings from code-implied bottlenecks because no `precompute.log` covers the full 120k.

### 2.3 Shared reusable infrastructure

**Principle:** Build platform pieces used across multiple use cases (models, features, observability) rather than copy-pasting per experiment.

**Repo mapping:**
- `grok_probe_qwen.py` / `modal_probe.py` / `modal_train.py` triple-own `HourglassProjector`, `make_collate`, `lr_at`, `ProbeMonitor`/`TrainMonitor`, `render_curves` — ~40% duplication, ~2500 lines. `modal_probe.py:124` even documents "ports of the validated grok_probe_qwen.py core" — intentional duplication.

**Cost if skipped:** The LayerNorm dtype bug had to be fixed **twice** (Colab CPU fp16 + Modal L4 bf16) because the same logic lived in two places.

---

## 3. From Hamel Husain / Shreya Shankar — Eval Discipline

> Sources: https://hamel.dev/blog/posts/evals-faq/index.html — https://www.sh-reya.com/blog/in-defense-ai-evals/

### 3.1 Single-variable ablations; log everything

**Principle:** Change one variable at a time. Capture the full trace (input → retrievals → agent steps → output), not just the final answer. Fix the earliest upstream problem first — later issues cascade.

**Repo mapping:**
- `modal_probe.py` ladder flags (`--no-grad-ckpt`, `--bucketing`, `--attn flex`, `--compile`, `--profile`) are the correct single-variable discipline — but they only landed on `refactor/discipline`. Before that, every run changed batch size + sequence length + streaming vs Volume simultaneously.
- `ProbeMonitor` / `TrainMonitor` log `loss/ema_loss/gnorm/lr/tok_s/samples_seen` — but not `git SHA`, `manifest hash`, `shard set`, or `seed` in the header. Runs are not comparable without re-reading code.

**Cost if skipped:** The 26s/step on L4 was attributed to "small model slow" until we measured that bucketing alone is 1.3–1.5× — because we never isolated it.

### 3.2 Spend 60–80% of dev time on evaluation, not building metrics

**Principle:** Review 20–50 outputs in a notebook with a domain expert before building automated judges. Use binary pass/fail, not 1–5 scales. Validate any LLM judge against human labels.

**Repo mapping:**
- No eval harness beyond `val_loss` on 2% held-out (`train_manifest_val.jsonl`). No pass/fail criteria for "does basic alignment work?" at 2k/200.

**Cost if skipped:** The minimal probe (2k/200) has no gate definition beyond "loss decreasing + gnorm finite" — correct, but not yet codified as a pass/fail check.

---

## 4. From Marin / TogetherAI — 4-Rung Scaling Ladder

> Source: https://github.com/marin-community/marin/issues/8435 — Percy Liang

Marin trained a **4-rung ladder from 1.6B-A61M (48B tokens) to 27.7B-A1.2B (926B tokens) to debug and forecast the 535B-A23B hero run (18.75T tokens, 11×GB200 NVL72, ~3 months, 2.7e24 FLOPs)** — at **1% of total compute**.

**Five stated purposes — each maps to our project:**

| Marin purpose | Vision-Adapter mapping | Cost if skipped |
|---|---|---|
| Generate performance forecasts; inferior outlook → redesign | Our L4 dryrun `loss=5.61 peak=12.32GiB` *is* the forecast for B300; without it, B300 is a blind bet | B300 OOM or grok failure after $100s |
| Monitor optimization behavior shifts with scale (grad magnitude, dropped tokens) | Our `n_vis` 35→4900 variance and `grad_norm` 28–205 bursts | Previous ladder revealed grad growth "to over 4" → added logit z-loss; later checks showed high-batch configs would "blow up mid-run" without it |
| Estimate eval trajectories over the full run to flag divergence | Our `samples_seen` grok window (57.6k ref) — step 3600 at bs16 vs 7200 at bs8 | Miss the cliff because wall-clock, not samples, was the axis |
| Small-scale baselines to interpret hero-run oddities | Our `grad_norm` climbing until 40% then decreasing with LR decay — hero-run pattern that looked alarming but was expected | Unnecessary intervention mid-B300 |
| Cost efficiency: 1% of total compute | Our L4 probe is <$1 vs B300's hundreds | The cheapest debug we have |

**Process lesson:** Each rung exists to kill a class of bugs *before* it compounds. We tried 0→304B with one duplicated probe. The ladder on `refactor/discipline` is the fix.

---

## 5. From Reproducibility Canon — PyTorch

> Source: https://docs.pytorch.org/docs/2.13/notes/randomness.html

**Completely reproducible results are not guaranteed** across releases, commits, platforms, or CPU vs GPU — even with identical seeds. Mitigations:

| Source of randomness | Control | Repo gap |
|---|---|---|
| Python RNG | `random.seed(0)` | `vision_adapter/data/dataset.py` now seeds all three RNGs (historical: monolith did this — but only this) |
| NumPy RNG | `np.random.seed(0)` + per-Generator seeding | Never set in manifest build |
| PyTorch RNG (all devices) | `torch.manual_seed(0)` | Never set in data pipeline |
| DataLoader workers | `worker_init_fn` seeding from `torch.initial_seed()` + `torch.Generator(g.manual_seed(0))` as `generator=g` | `num_workers=8` in production loader has no `worker_init_fn` |
| cuDNN benchmarking | `torch.backends.cudnn.benchmark = False` | Not set |
| Deterministic algorithms | `torch.use_deterministic_algorithms(True)` — raises if op has no deterministic impl (e.g. `index_add_cuda`) | Not set; SDPA fused backends are non-deterministic in backward — `FLASH_ATTENTION`/`EFFICIENT_ATTENTION` differ by accumulation order |
| Uninitialized memory | `torch.utils.deterministic.fill_uninitialized_memory` (True when determinism on) | Not considered |

**Tradeoff:** Deterministic selections run slower — use for debugging/testing, not for the timed B300 run.

**Concrete fix for Vision-Adapter:**

- Add `random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)`, and `worker_init_fn` in `build_train_manifest` and `_shared_setup`.
- Add `ORDER BY image` to `vision_adapter/data/dataset.py: ORDER BY image LIMIT 54000` — the seed does not save you if the input order is nondeterministic (historical: monolith) if the *input* row order is nondeterministic.
- Pin MoonViT repo with `revision=` in `hf_hub_download` (`moonvit.py` / `vision_adapter/data/dataset.py` + `vision_adapter/models/precompute.py` (historical monolith — see PIPELINE.md)).

**Cost if skipped:** Two rebuilds of the same 120k pick different 54k agentic rows — the grok window measurement is not comparable across runs for reasons unrelated to the model.

### 5.1 From Inference Engineering — Roofline Model (Ch 2.4.1/2.4.2)

> Source: `/home/keypaa/Downloads/Inference Engineering.pdf` — Inference Engineering, Philip Kiely (Baseten Books, Dec 2025, 978-8-9943597-2-3) — Ch 2 *Models*, §§2.4.1–2.4.2, Figs 2.12–2.13, text-extracted to `/tmp/inf_eng.txt` (9085 lines, 50-image cap avoided via `pdftotext -layout`).

**Ops:Byte ratio & arithmetic intensity — why it picks your optimization:**

- A GPU advertises **compute (ops/s)** and **memory bandwidth (GB/s)**. Their ratio is the **ops:byte ratio** (e.g. H100 FP16 989 TFLOPS / 3.35 TB/s ≈ 295). A kernel's **arithmetic intensity** = ops performed ÷ bytes moved. If intensity < ops:byte, the kernel is **memory-bound** (waiting on bytes); if > ops:byte, **compute-bound** (waiting on FLOPs). The **roofline model** (Fig 2.13) plots performance vs this diagonal — below the roof you are memory-bound, on the flat top you are compute-bound.

**Repo mapping — $26\,s/step @ bs16$ on L4:**

- `modal_probe.py:500` `train_step` with `bytes:ops` is where the roofline test applies — same `26s/step` that drives `TRAINING_PLAN.md Phase 1` and the `26\,s/step is huge?` discussion. Our `peak=12.32\,\text{GiB} / 22\,\text{GiB}` measured in the L4 dryrun sits well below the memory roof — the run is **not memory-bound**, so the winning lever is **kernel speed** (fewer ops or better fused kernels), not more quantization. That is why **A1 FlexAttention > INT4** for the probe when not memory-bound: Flex is an *implementation improvement* that reduces ops without changing numerics; INT4 only helps if you are on the memory-bound side of the roofline.

**Cost if skipped:** You would chase INT4/MXFP4 for the 2B probe because "quantization helps inference" in the abstract — but at `12/22` GiB you would save no step time, only add Hamming-style numerics debugging. The roofline tells you to fix `head_dim=512 → eager` first.

### 5.2 From Inference Engineering — CUDA Kernel Selection & Fusion (Ch 4.1.2–4.1.3)

> Source: same book, Ch 4 *Software*, §§4.1.2–4.1.3, Figs 4.1–4.2.

**Kernel selection — hardware-specific, not portable:**

- `§4.1.2` *"A kernel written for an H100 will likely fail to take advantage of the architecture and extra memory of a B200, while a kernel written for that B200 could be backwards incompatible with Hopper … With each generation of GPUs, porting handwritten kernels to run optimally takes substantial engineering work."* Most selection is automatic (PyTorch/TensorRT-LLM), but **manual insertion of a GEMM kernel is the exception** — the book's example is `DeepGEMM` for **FP8 GEMM on Hopper out of DeepSeek-V3** (precise matrix dims) — exactly our `modal_train.py:402` FP8/FP4 path where we swap `transformers.integrations.finegrained_fp8` → `Max` with `_fp8_linear_train`, and where `kernels>=0.16` fetches the hub kernel at runtime.

**Repo mapping:**

- `modal_train.py:402–436` `_fp8_linear_train` / `Max` monkey-patch (uses `cutlass` under the hood) — the plugin boundary `§4.1.2` warns about.
- `modal_train.py:439-501` `_make_chunked_eager` chunking `budget 2**26` — a hand-fused eager attention that trades `1.4×` compute for working inside the memory ceiling (Fig 4.2 logic: back-to-back kernels waste `write→read` round-trips).
- `modal_probe.py:260` / `grok_probe_qwen.py:801` `train_step` hot loop — `torch.compile` (Ch 4's compilation step) fuses long lightweight kernel sequences, but **cannot fuse `DeepGEMM`/`FlashAttention` plugin kernels**. Useful for our non-DeepSeek parts (long `LayerNorm+GELU` chains in `HourglassProjector` at `modal_train.py:148` vs `grok_probe:165`), not for the FP8 path.

**Kernel fusion — why it matters differently for prefill vs decode:**

- `§4.1.3` The book's toy example `multiply_by_2 → save → load → multiply_by_3` fused into `multiply_by_6` eliminates the middle round-trip — **during decode, the bandwidth-bound phase, fusion matters most** because an engine "can't afford unnecessary reads/writes." During **prefill** (our training forward), the system is compute-bound, so fusion saves fewer bytes but still removes kernel-launch overhead. Our `precompute` `ThreadPoolExecutor(8)` hides CPU decode time, but GPU-side fusion is separate.

**Cost if skipped:** On B300/SM103, landing FP8 without verifying the `torch==2.13.0 / cu130 / sm_103` kernel actually fuses (and that `DeepGEMM` Blackwell support is at "now supported" per the book's §4.1.2 note) would be a silent 1.3–1.5× left on the table — and on a multi-day run, that is days.

**Status of this source in the repo:** Not generic advice. File:line-bound above for `modal_train.py:402`, `modal_probe.py:260`, `grok_probe_qwen.py:801`, and `TRAINING_PLAN.md Phase 5` B300 stack. The book's `§4.1.2` DeepGEMM paragraph is the exact pattern our FP8 patch follows.

### 5.3 From Inference Engineering — Image & Video Generation (Ch 6.5)

> Source: same book, Ch 6 *Modalities*, §6.5, Figs 6.9. Text-extracted (50-image cap avoided via `pdftotext -layout`); remaining chapters re-extracted after reboot to `/tmp/inf_eng.txt`.

**The transferable point is not image quality — it is attention + GEMM discipline under modality-specific budgets:**

- `§6.5.1` High-performance image inference uses **FlashAttention 2 → 3 → 4 by GPU generation** (Hopper → Blackwell), plus fused `RMSNorm` kernels and **GEMM quantization to FP8 for 2× Tensor Core FLOPS** (`CuTe / CUTLASS / DeepGEMM` selection model-by-model). **Torch compilation fuses the long tail of lightweight kernels and is cached for faster node startup** (compilation takes minutes, so cache it — `modal_train.py:402` FP8 path excepted, same caveat as §5.2). The book's line that image generation is "theoretically compute bound" but *still* needs memory-efficient kernels to *reach* that bound is the same lesson as our `35→4900 n_vis` precompute: varlen packing is a memory prerequisite before the compute bound matters.

**Repo mapping:**

- `moonvit.py:118-226` varlen `flash_attn_varlen_func` vs fallback `scaled_dot_product_attention` loop — the `FA2/FA3/FA4` per-generation upgrade path at `vision_adapter/models/moonvit.py:74-85` (wheel pin `flash_attn-2.7.4`) and the `precompute_bench` in `vision_adapter/models/precompute.py` (historical: monolith) that sweeps `patch_cap` vs `peak GiB / util%` are Ch 6.5's "pick FA by architecture" in practice. Our `MAX_PATCHES=65536`/`MAX_SIDE=7168` capping is the `n_vis`-level equivalent of Ch 6.5's quantization granularity — keep it, but measure it (we already do: `test_preprocess.py:22-42`).
- `TRAINING_PLAN.md Phase 3` skipped `ParquetEmbSFT` because `.pt` vs parquet was publish-quality not speed — the book's Ch 6.5 closing note that `vLLM / SGLang Diffusion / TensorRT` vs `PyTorch` is an *image-model-specific* choice explains why a training-time `ParquetEmbSFT` would not have helped: our `EmbSFT` warm `7ms` `torch.load` is the PyTorch-side win the book attributes to `__CUDA_VISIBLE_DEVICES` vs engine choice.

**Cost if skipped:** Upgrading GPUs (T4→A100, A100→B300) without re-picking `FA2→FA3→FA4` per generation and without cached `torch.compile` on the B300 pipeline leaves the same 1.3× on the table — at `modal_train.py:439` `_make_chunked_eager` the hand-fused chunk budget `2**26` already trades `1.4×` compute for the memory ceiling (Fig 4.2 logic), so the *next* 1.3× must come from FA generation, not more chunking.

**Reminder:** No section of *Inference Engineering* (hardware Ch 3, local inference §3.5, Ch 4 software, Ch 6.5 image-gen, Ch 7 production) contains the GLM 5.2 SFT recipe — that remains the Baseten blog + `baseten/GLM-5.2-Vision-NVFP4` (§1.4). This book is a serving book; its value here is hardware/roofline/kernel-plumbing, not the SFT recipe.

---

## Checklist — Ordered Cheapest→Highest-Leverage (each = file:line edit on THIS branch)

- [ ] **Add `ORDER BY image` to agentic `LIMIT 54000`** — `vision_adapter/data/dataset.py: ORDER BY image` (historical: `modal_pipeline:1081`) — one-line determinism fix; unblocks local-vs-Modal parity.
- [ ] **Pin MoonViT `revision=`** — `vision_adapter/models/moonvit.py` / `vision_adapter/models/precompute.py` `hf_hub_download(..., revision=...)` (historical: monolith did this without pin) — one kwarg; prevents silent model change on force-push.
- [ ] **Keep `compression=None` everywhere** — `vision_adapter/data/pack.py:compression=None` (historical: monolith at 606 was missing; matches staged pack) — already measured 2.3× win on BF16.
- [ ] **Retain `patch_sizes.json` on Volume** — `vision_adapter/models/precompute.py` retains `patch_sizes.json` (historical: monolith at 1139) — saves ~10 min cold header reads per precompute.
- [ ] **Sort/bucket by `n_vis` before greedy packing** — `vision_adapter/models/precompute.py` / `vision_adapter/models/precompute_colab.py` (historical: monolith) — one `sorted(cache_items, key=...)` removes 10–20% fragmentation, zero numerical risk.
- [ ] **Re-pick `patch_cap` via `precompute_bench` on A100** — `vision_adapter/models/precompute.py` default `262144` (historical: monolith at 1250) is T4-tuned; A100 can carry 350–524k at <85% mem.
- [x] **Single config object** — DONE on `refactor/discipline` via `vision_adapter/config.py:TrainConfig` + `config_header` (replaces scattered LR/BATCH_SIZE/WARMUP). — new `configs/qwen_probe_small.yaml` + `configs/qwen_probe_full.yaml` dataclass — replaces scattered `LR/BATCH_SIZE/MAX_SEQ_LEN/WARMUP` at `grok_probe_qwen.py:104`, `modal_probe.py:65`, `modal_train.py:49`.
- [x] **Add manifest header** — DONE via `vision_adapter/manifest.py:write_manifest_with_header` (header-first, `manifest_version=1`, `git_sha`, `seeds`, `upstream`, `shard_set_hash`). — `vision_adapter/manifest.py:write_manifest_with_header` (historical: monolith at 1115 write path) — `manifest_version, git_sha, seeds (py/np/torch), upstream revisions, timestamp, row count, shard-set hash`.
- [x] **Add per-shard `sha256(emb_XXXX.parquet)`** — DONE via `vision_adapter/config.py:file_sha256` + `vision_adapter/data/pack.py` per-shard hash (Volume↔HF parity). — `vision_adapter/data/pack.py` + `vision_adapter/config.py:file_sha256` post-write (historical: monolith at 562) — Volume↔HF parity.
- [x] **Experiment registry + enriched logging** — DONE via `vision_adapter/registry.py` (`runs.jsonl`, `registry_entry`) + `vision_adapter/config.py:config_header` + `run_end` (`run_id`-correlated). — `modal_probe.py`/`grok_probe_qwen.py` JSONL header: `run_id, git SHA, manifest hash, shard set, seed, device/dtype, step_ms breakdown`. One registry file for `Configuration | sec/step | samples/sec | tokens/sec | VRAM | Relative`.
- [x] **Extract `src/training/` shared core** — DONE via `vision_adapter/core.py` (`HourglassProjector`, `make_collate`, `train_step`, monitors) shared by all trainers. — `HourglassProjector`, `make_collate`, `embeds_for`/`visual_inject`, `train_step`, `ProbeMonitor`, `render_curves` — single source for `modal_train.py` + probes.
- [ ] **Minimal probe gate codified** — 2k/200, fixed batch, single seed, `loss ↓ + gnorm finite` pass/fail — `modal_probe.py` with `--sample-size 2000 --max-steps 200 --bucketing` ladder flags.

---

## Anti-Patterns We Already Hit

| Anti-pattern | What happened | File:line | Fix |
|---|---|---|---|
| `modal.Mount` deprecated | `modal_probe.py` first dryrun raised `AttributeError: module 'modal' has no attribute 'Mount'` | `modal_probe.py:102` (fixed to `image.add_local_file`) | Use `image.add_local_file("modal_train.py", "/root/modal_train.py")` |
| Mixed-dtype LayerNorm | `EmbSFT` `float32` vis → bf16 projector `LayerNorm` raised `expected scalar type Float but found BFloat16` | `modal_probe.py:137` / `grok_probe_qwen.py` same | Cast `vis.to(proj_dtype)` before `proj()` |
| 67M vs 67B terminology | Projector 67,129,344 params confused with DeepSeek 304B/67B active | Log line at `modal_train.py:9` / `ARCHITECTURE.md:23` | Every log now says "67M projector" explicitly |
| `LIMIT` without `ORDER BY` | Agentic 54k selection nondeterministic; seed doesn't save it | `vision_adapter/data/dataset.py: ORDER BY image` (historical: `modal_pipeline:1081`) | Add `ORDER BY image` |
| No `revision=` pin | MoonViT repo force-push would silently change model | `moonvit.py` `hf_hub_download` | Pin to commit hash |

---

*Generated on `refactor/discipline` from `master@3cb0d6f`. Not generic advice — every item maps to a file:line above or is dropped.*
