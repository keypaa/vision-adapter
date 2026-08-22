# HANDOFF — Vision-Adapter (cold-start for the next harness)

> Read this plus `docs/TRAINING_PLAN.md` (authoritative phase record).
> Repo: `keypa/vision-adapter` (public, branch `master`, remote `origin`).
> Working tree is **clean** as of this handoff. Tests: **41 passed**.

## TL;DR

Embedding pack is done and published (103/103 shards on HF). The training
gate moved to the **B300 dryrun** — the A100 gate was **abandoned by owner
decision**: A100 (cc 8.0) has no FP8 GEMM, so transformers force-dequantizes
the checkpoint to bf16 (~310 GiB), which contradicts the owner's requirement
to keep weights in native FP8/FP4. On B300 (sm103) the backbone loads native
FP8 (~155 GiB, all-in-VRAM, fits with ~110 GiB headroom). Three B300-gate
fixes landed after the handoff below was written — read "V4 gotchas" before
touching `modal_train.py`.

```
[✅] images repo 17/17 ── [✅] manifests ── [✅] embeddings 103/103 on HF
        │
        ▼
[▸ NEXT] modal run --detach modal_train.py::train_dryrun_b300  (Phase 5 gate)
                │   fixes so far: kernels pkg, embed-hook injection,
                │                 chunked eager attention (commits abe4f51,
                │                 4df3137, 1065af3)
                ▼
        MEMORY GATE PASS + ckpt ON/OFF verdict in dryrun_report.txt
                │
                ▼
        [train_b300]  (~30k steps, watched via train_curves.png)
```

## Snapshot (numbers re-verified at handoff time)

| Item | Verified value |
|---|---|
| HF `vision-adapter-images` | **17/17 shards, 138,987 rows** |
| HF `vision-adapter-manifests` | **5 files**, complete |
| HF `vision-adapter-embeddings` | **103/103 shards** (104 files = 103 + README), spot-checked |
| Spot-check | `emb_0050` (1360 rows, 22 RGs) — 3 random rows `vis_bytes→bf16` **ROUND-TRIP PASS** vs volume `.pt` |
| Modal volume `embeddings/` | **138,987 `.pt`** intact, never deleted |
| Modal volume `shards/` | **empty** (18 GB smoke shards deleted after HF verification) |
| Local HF cache `~/.cache/huggingface` | **purged to 368 KB** (was 20 GB) — re-downloads on demand |
| Local staging `/var/tmp/emb_stage` | `pack_progress.jsonl` + `.png` only (60 KB, 21 GB peak never accumulates) |
| `torch` locally | `2.13.0+cu130` — cu128 wheels are **broken on B300/SM103** |

## What was built this session (commits on `master`)

| Group | Commits | Note |
|---|---|---|
| Packer | `cf47941`..`998dbd4` (local_pack.py) | streaming writer, resume semantics, retry, `-hf-only`, `-workers`, background pipeline, progress, `pack_progress.*` |
| Perf | `356696c` | `compression=None` (2.3× faster writes), pipelined download↔HF-push |
| Telemetry | `2d9ebaf` | `TrainMonitor` (EMA/median + spike), `render_curves`, enriched `_one_step` (gnorm), live `train_curves.png` |
| Data contract | `8363f5f` | **fixed real bug** in `make_collate` (answer trimmed to 1 token on long prompts — now keeps answer, trims user); `test_train_collate.py` |
| Trainer hardening | `456133d`, `b36d9b8`, `ebf91a1`, `6e1dfad`, `998dbd4`, … | startup heartbeats, B300 image/stack, resume fix, `pin_memory`, ckpt ON/OFF measurement, timers |
| Cleanup | `fc00636`, `ef3825f` | stage-dir → `/var/tmp` (tmpfs would ENOSPC), size-verified downloads |

Relevant docs/specs/plans (all tracked):
`docs/TRAINING_PLAN.md` (updated: Phase 2 IN PROGRESS via local_pack, Phase 3 SKIPPED by measurement, Phase 4 runs `EmbSFT`, Phase 6.1 DONE, Phase 7 implemented)
`docs/superpowers/specs/2026-08-18-local-pack-design.md`
`docs/superpowers/plans/2026-08-18-local-pack.md`

## How to continue (copy-paste)

Every new terminal:
```bash
cd ~/Projects/DSV4-0731/Vision-Adapter
source .venv/bin/activate
```

The pack is done — verify HF if you want, then dryrun:
```bash
# verify HF (expect 103)
python -c "from huggingface_hub import HfApi; print(len([f for f in HfApi().list_repo_files('keypa/vision-adapter-embeddings',repo_type='dataset') if f.endswith('.parquet')]))"

# A100 gate — records step-time baseline the B300 dryrun compares against
modal run --detach modal_train.py::train_dryrun
modal volume get vision-adapter-data dryrun_report.txt . && cat dryrun_report.txt   # must contain PASS

# B300 gate (cu130, all-in-VRAM, ckpt ON/OFF measured back-to-back)
modal run --detach modal_train.py::train_dryrun_b300
modal volume get vision-adapter-data dryrun_report.txt . && cat dryrun_report.txt

# full training
modal run --detach modal_train.py::train_b300      # or ::train for the A100 fallback
modal volume get vision-adapter-data logs/train_curves.png . && open train_curves.png
```

Single-instance guard on the packer: `flock -n /var/tmp/local_pack.lock` — don't launch two packs concurrently.

## Visual feedback (audited + fixed)

| Program | Startup | During run | Durable artifact |
|---|---|---|---|
| dryrun | timed phases (`tokenizer`, `loading backbone (~155 GiB — minutes on cold)`, `datasets ready`) | end line: `MEMORY GATE PASS/FAIL | step=Xs (Y it/s)` | `dryrun_report.txt` (both ON/OFF lines on B300) |
| training | same startup heartbeats | every 20 steps: `loss/ema/gnorm/tok-s/ETA/ALERTS`; `[val]` every 250; `[SPIKE-ALERT]` | `logs/train_log.jsonl` per step + `logs/train_curves.png` every 250 |
| local_pack | header `embeddings/shards/workers` | intra-shard `staging … files (GB, MB/s, s)` every ~3 s + `packed …/1360 rows` + `wall Xs (stage Ys pack Zs push Ws)` | `pack_progress.jsonl` + `.png` in stage dir |

The B300 path is **fully staged** and decoupled from the pack work:
- `train_image` (A100): `torch==2.5.1` + `device_map="auto"` (70 GiB GPU / 200 GiB RAM)
- `train_image_b300`: `torch==2.13.0` (PyPI default = **cu130**, SM103 verified — cu128 broken on B300 per pytorch#175842) + `device_map={"": 0}` all-in-VRAM, `B300_GPU_MEM_CAP_GIB=250`

## Locked decisions / gotchas for the next harness

### DeepSeek-V4 trainer gotchas (learned 2026-08-22, B300 gate runs)

- **Vision injection must use the `visual_inject` embed_tokens hook** (commit
  `4df3137`). V4's hash-MoE gates route experts via a frozen
  `tid2eid[input_ids]` lookup and the core model *raises* if `input_ids` and
  `inputs_embeds` are both passed — plain inputs_embeds injection is
  structurally impossible. The hook splices projector output into embedding
  OUTPUT rows; ids stay the model input.
- **V4 attention is eager-only**: `_supports_sdpa = False`; FA2/vLLM-FA3
  documented incompatible (sparse attn + mHC streams). Eager materializes
  `[B, heads, L, L+1]` logits in **fp32** ≈ 46 GiB at bs=8 → OOM beside the
  FP8 backbone. `_patch_chunked_eager_attention()` (commit `1065af3`) chunks
  query rows — mathematically identical (verified vs reference in
  `test_train_collate.py`), drops attn_weights (output_attentions=False).
- **transformers' finegrained-fp8 needs `kernels>=0.16,<0.17`** in the image
  (`abe4f51`) — it fetches the GEMM kernel from the hub at runtime.
- **Native FP8 residency works on B300 only**: cc<8.9 (A100) makes
  transformers dequantize to bf16 (~310 GiB) — that is why the A100 gate was
  abandoned. If an A100 fallback is ever needed, it requires a custom
  FP8-resident JIT-dequant loader (owner has not asked for this).
- Each dryrun costs ~$0.30 — verify locally (AST + pytest + equivalence
  tests) before every launch; the suite caught two real bugs pre-launch.

- Parquet is the **publish/reproduction format only**; trainer stays on `EmbSFT` reading `.pt` (Phase 3 SKIPPED — warm reads 16 ms/file = 0–1% of a step, `num_workers=8` now applied; `pin_memory=True`).
- `--hf-only` = HF is the destination; `/data/shards` copy is optional insurance. Fixed resume bug: `resume_action(..., hf_only=True)` skips HF-present shards (previously redid every shard on reruns).
- `/tmp` is **tmpfs (11 GB, 2.5 GB free)** — NEVER stage there; `local_pack --stage-dir` defaults to `/var/tmp/emb_stage` (disk, 151 GB free).
- B300 dryrun decides grad-checkpointing ON/OFF on measured numbers (`ckpt=OFF KEEP OFF / TOO HOT`), not estimates.
- `cli workers=ANY` arrives as a string on Modal — coerce in-function; every `print` in Modal fns must use `flush=True`.
- `local_pack` peaks at **~21 GB disk** (parquet + next shard's staged `.pt`), never accumulates across 103 shards.
- GitHub: `https://github.com/keypaa/vision-adapter` (public, `master`). Push via `git push origin master`.

## Test suite (run before touching code)

```bash
source .venv/bin/activate
python -m pytest -q    # 36 tests: test_preprocess (5), test_local_pack (19), test_train_telemetry (7), test_train_collate (5)
```

## Remaining in strict order

1. A100 dryrun (Phase 4) — PASS + baseline speed
2. B300 dryrun (Phase 5) — PASS + ckpt verdict vs A100 baseline
3. Full `train_b300` (~30k steps) — watched via telemetry
4. Post-training only: delete `/data/embeddings/*.pt` (~950 GB recovery)
5. Polish: dataset cards + HF Space linking the 3 repos (optional)

## If anything interrupts

All three long runners are resumable: rerun the same command — finished shards/steps skip instantly. For the pack, finished HF shards are skipped; for training, checkpoints resume. Never launch two packs at once.

---
Handoff written: 2026-08-22. Last verified commit at handoff: `998dbd4` (+ local unpushed fixes `1dcdf3e`, `6e1dfad` etc. — `git log origin/master` to confirm).
