# NEXT_STEPS — post-probe roadmap (consolidated 2026-09-08)

Single source of truth for what remains after the Qwen3.5-2B probe.
Includes the surviving recommendations of both fresh-eyes reviews
(adaptive-ckpt reviewer = R1, FlexAttention researcher = R2).

## 0. Where we stand (done)

- [x] Bucketed HF streaming, volume deleted, repack 103/103 verified
- [x] L4 probe 200 + Molab PRO 6000 probe 1000 (plateau, no grok yet — expected)
- [x] OOM root cause on worst bucket (4901+, 1.8–2.9GiB RGs, uncapped collate L)
- [x] Per-batch adaptive grad-ckpt on B·L² budget (`a8ff94e`, 79 tests green)
  - trigger: shape-gated (B·L²), NOT mask-sum (R1 hole 3)
  - direct flag flip + try/finally, no hook leak (R1 holes 1–2)
  - `expandable_segments` default (R1 extra)
  - threshold L_MAX=2500 / COST_MAX=50M, env-overridable
- [x] GPU gate PASSED: 50-step FORCE_LARGEST_BUCKET repro survives worst
  shards, loss 11.4 → 3.6, ckpt fires (R1 hole 6 closed: ckpt suffices,
  failure was retained activations, not kernel workspace)
- [x] HF checkpoint push (ckpts + probe_log each save, manifest/log/curves at end)
- [x] Stage-timed stream logs (`97bf267`: load_span announce, metadata line, HEAD timeout)

## 1. Tonight — probe 4000 (owner: Molab session)

```bash
git pull  # >= 97bf267
pip install -e ".[train]" && mkdir -p data
export HF_TOKEN=... VISION_ADAPTER_PUSH_HF=1
export VISION_ADAPTER_HF_CKPT_REPO=keypa/vision-adapter-probe-checkpoints
nohup python -m vision_adapter train --data-dir ./data --config probe \
  --max-steps 4000 --dtype auto \
  --push-to-hf --hf-ckpt-repo keypa/vision-adapter-probe-checkpoints \
  > data/train_4000.log 2>&1 &
```
- Keep the tab open (container follows the tab; nohup ≠ session-proof).
- First HF push at step 500 (~40min). ETA ~7h (worst bucket ≈ +2h).
- Success = reach step 3600+ (57.6k samples, grok window) and inspect loss collapse.
- Afterwards: `plot_probe.py` graph + heldout-60 eval on final ckpt.

## 2. R1 leftovers (adaptive-ckpt reviewer)

- [ ] **Smoke-fallback crash** (separate bug, small): after any streaming
  failure, fallback crashes with `FileNotFoundError` (fetched manifest never
  saved locally). Fix: persist fetched manifest to `data_dir`, guard fallback.
- [ ] **Long-term fix: token/L²-budget batch sampler** (R1 ranked #1):
  split homogeneous huge-n_vis runs into smaller micro-batches + grad
  accumulation. Only if monster-step cost still hurts after Flex.
- [ ] Optional: selective ckpt on attention layers only (less slowdown than
  full ckpt on monster batches).
- Rejected per R1: `expandable_segments` as sole fix (done anyway, ~10%),
  always-on ckpt (too slow), silent L-cap in collate (drops signal).

## 3. R2 plan — attention speedup (Flex / FLA)

Context: monster batches run 40–85s/step (ckpt ON, L 5.4–9.6k, B=16).
Normal batches ~1.6–5s. Padding + attention dominate.

### 3.1 Measure first — micro-bench (decides everything)

Harness (Molab, CUDA, frozen Qwen3.5-2B bf16, synthetic inputs_embeds + 2D masks):
shapes (B=16, L in {2k, 5.4k, 8k, 9.6k}) × patterns
{uniform r=L, bucketed r≈0.7L, adversarial r≈0.25L}.
10 timed steps after 3 warmup (never step 0), nvtx ranges + profiler,
`max_memory_allocated`.

1. **Apportionment (gate)**: per-layer-type CUDA time (full vs linear vs rest)
   at (16, 8k) adversarial, ckpt ON. Attribute ≥80% of step time.
   - linear ≥50% → prioritize Option 0 below.
2. **SDPA backend probe**: `TORCH_LOGS=sdpa` / kernel names — math vs
   mem-efficient (R2 suspects math fallback: 15–30GiB materialized/layer).

### 3.2 Options (ranked by R2)

- **Option 0 (try first): `pip install flash-linear-attention causal-conv1d`.**
  Zero modeling changes; swaps all 18 linear layers to fused kernels.
  Risk: sm100/s
...[truncated 3075 chars]