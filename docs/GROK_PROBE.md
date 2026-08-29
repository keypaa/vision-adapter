# GROK_PROBE — rung 1 of the validation ladder

`grok_probe_qwen.py` trains the HourglassProjector against a small text-only
Qwen3.5 backbone using the **existing** cached MoonViT-V2 embeddings and train
manifest (subsampled). Purpose: prove the full recipe end-to-end on cheap
hardware AND timestamp the grok collapse in `samples_seen` before committing to
the DeepSeek-V4-Flash B300 launch.

## What it runs

| Piece | Source | Note |
|---|---|---|
| Vision features | `keypa/vision-adapter-embeddings` (103 parquet shards, **884 GiB total**) | **streamed over HTTPS row-group by row-group (~400 MiB each) — nothing bulk-downloaded**; reconstructed `frombuffer(uint8) → bf16 → reshape(-1, 4096)` exactly as `test_local_pack.py` pins; the vision tower is never executed |
| Manifest | `keypa/vision-adapter-manifests/train_manifest.jsonl`, seeded global-random subsample (`--sample-size`) | subsample size printed at startup — normalize grok windows per samples_seen |
| Projector (only trainable piece) | `LN(4096) → Linear(4096, 2·H) → GELU → Linear(2·H, H)`, H = Qwen's actual `hidden_size` read from config at runtime | params: **25,180,160** @ 2B (H=2048), **34,094,592** @ 4B (H=2560) |
| Injection | `inputs_embeds` ONLY — Qwen3.5 raises on ids+embeds together ("specify exactly one"); visual span overwrites rows `[1 : 1+n_vis]`; `<\|image_pad\|>` sentinel exists (id 248056) but stays unused; positions masked out of loss like modal_train | grads flow through the spliced rows to the projector |
| Loss masking | `-100` everywhere except answer + EOS; ported invariant checks asserted on the first batch | same contract as `modal_train.make_collate` |
| Precision | bf16 on Ampere+ (`auto`), fp32 on T4 — **no quantization anywhere** | clean grok signal |
| Optimizer | AdamW(0.9, 0.95) wd=0, lr 5e-4, warmup → cosine to 10%, clip 1.0 | |

## Data plane: how streaming works (verified against the live repo)

- Each shard's **key index** is built remotely from footer + key-column-chunk
  range reads only (~1–2 s/shard, ~103 s for all shards) — this happens BEFORE
  any big transfer so the seeded subsample knows exactly which shard holds
  each of its rows.
- During training each needed **row group (~64 rows ≈ 330–450 MiB)** is
  fetched with 4 parallel HTTP range streams (~8 s at ~55 MiB/s) into RAM,
  decoded by pyarrow locally, then released. RAM peak ≈ one decoded group +
  one in-flight fetch ≈ **~1 GiB**.
- Delivery order is **shard-major** (the one concession to streaming); the
  random subsample itself is global-seeded, identical statistics to
  modal_train's shuffle. Deterministic given `--seed`, so `--resume`
  fast-forwards exactly.
- The two old Modal smoke shards (`emb_0000/0001`, single 1360-row groups
  ≈ 9 GiB spans) are excluded from the plan: unstreamable within a Colab RAM
  budget; they cover ~2 % of the corpus.

## Run on Colab (free tier)

```python
# 1. pip install line (fresh VM):
!pip install -q "transformers>=5.12" datasets safetensors huggingface_hub pyarrow matplotlib numpy

# 2. fetch grok_probe_qwen.py from the repo, then:
!python grok_probe_qwen.py --model qwen2b --sample-size 20000 \
    --batch-size 8 --max-steps 5000 --resume

# Optional but recommended on Colab: put your HF write token in Secrets as
# HF_TOKEN so checkpoints survive crashes (see below). No token is needed to
# READ anything — all data repos are public.
```

Notes:
- On a free **T4**: fp32 automatically (no bf16 silicon); try `--batch-size 4`
  if OOM. Free-tier **A100**: keep bs 8 or try 16.
- `--limit-layers 4` smoke-tests the wiring in minutes before the long run.

## Crash resilience

Every `SAVE_EVERY=500` steps the script pushes a bundle to
`keypa/vision-adapter-grok-probe`: `latest.safetensors` + `latest.opt.pt`
(projector + AdamW state incl. step count) + `probe_log.jsonl` +
`probe_curves.png`. A dead VM loses at most 500 steps. Rerun with `--resume`:
local checkpoints are used first; if absent (fresh VM), the bundle is pulled
from HF, the LR schedule is recomputed at the restored step, and the streamed
data fast-forwards deterministically past all consumed rows. Without
`HF_TOKEN` set, pushes degrade gracefully to local-only checkpoints (a warning
is printed).

## Expected wall-clock

Backbone fwd+bwd dominates each step; streaming (~8 s per 64-row group =
~8 steps' worth of data at bs 8) overlaps well once warm.

| GPU | dtype | est. s/step @ bs8 | 5000 steps | + startup (index+model) |
|---|---|---|---|---|
| Colab T4 16 GB | fp32 | ~1.5–3 s | 2.5–4 h | ~15 min |
| Colab A100 40 GB | bf16 | ~0.3–0.6 s | 30–60 min | ~10 min |

Steps-to-one-epoch at bs8 = **2500** (20k samples); default 2 epochs = 5000
steps ⇒ samples_seen ceiling 40k.

## What to send back

1. `probe_log.jsonl` — per-step JSON lines `{step, loss, ema_loss, gnorm, lr,
   tok_s, samples_seen, elapsed_s}` (+ `run_end` summary).
2. `probe_curves.png` — re-rendered every 50 steps; raw loss + EMA + LR with
   x-axes in BOTH steps and samples_seen. Also auto-pushed to the HF bundle.
3. Final loss / final EMA (printed in the DONE line).
4. `samples_seen` at any observed collapse — an explicit
   `*** COLLAPSE *** step=N samples_seen=M` banner fires when the recent mean
   falls below half the window median.

## Honest scope note (read before interpreting a null result)

The Baseten reference collapse is ~57.6k samples seen. At bs8 the 20k×2-epoch
plan tops out at **40k samples_seen — BELOW the reference window**, reached
partly on replayed data after epoch 1. So:

- **Collapse observed** → strong positive signal, extrapolate directly.
- **No collapse by run end** → ambiguous: could be "needs more samples"
  (confounded by unique-data starvation) rather than "recipe broken".
  Follow-up = rerun on Modal/L4 with the full 120k manifest (~$1–2 on GPU)
  where samples_seen can cross 72k+ on mostly unique data.

While loss is flat you will see periodic plateau banners explaining this is
the expected grok phase — do not restart because the curve looks dead.
