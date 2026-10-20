# REFERENCE RECIPE — what the grok actually is, and how far our probe is from it

Sources (primary, fetched 2026-10-20):

- **Harry Partridge, "GLM 5.2 With Vision"** (X article, 2026-07-16),
  `x.com/part_harry_/article/2077610277571637435`.
- **0xSero, "Teach GLM-5.2 to see art"** (`github.com/0xSero/glm-vision`),
  the reproduction: `training/train_sft.py`, `training/data.py`, plus the dataset
  card of `keypa/vision-adapter-manifests` (0xSero's own mix).

Harry is a Baseten engineer, so "the Baseten recipe" and "Harry's recipe" are
the same thing. 0xSero's repo is the *reproduction*; his `glm-vision-sft-mix`
is the *data* we ingested.

---

## 1. The reference recipe, exactly

| | reference (Harry) | 0xSero's reproduction | **our probe** |
|---|---|---|---|
| data | 66k images + short QA, 2 epochs | 37k art images, 2 epochs | 117.6k rows (agentic/doc/conv) |
| batch | **64** | **64** (microbatch 4, grad-accum 16) | **16** |
| LR | **5e-4** | **5e-4, CONSTANT** (no warmup, no cosine — `AdamW(lr=...)` + `opt.step()`) | 5e-4, **warmup 100 + cosine to 10%** |
| weight decay | — | 0.0 | 1.0 (TrainConfig) |
| grad clip | — | 1.0 | 1.0 |
| loss | CE on answer tokens only | CE on answer tokens only | CE on answer tokens only ✓ |
| image budget | — | **`max_image_pixels = 300_000`** → ≤**383** merged tokens (300k px ÷ 28² px/token) | **up to 16,598** tokens (mean 685) |
| seq budget | — | `max_seq_len = 2048` | 4,096 |
| answers | "simple question/answer pairs rather than long descriptions … our SFT dataset did not have any reasoning" | "deliberately SHORT (single actions / single phrases) — the recipe found short on-policy answers are required" | mean 2.0 words ✓ (0.55% long → being dropped) |
| backbone | GLM-5.2 (504B, NF4) | GLM-5.2 | Qwen3.5-2B |
| projector | 50M, 2-layer MLP | 44.8M | 67.1M (4096→8192→4096) |

**The 300k-pixel cap is 0xSero's builder rule, and it is the same rule the
source dataset card states** ("resize to <=300k pixels, dimensions rounded to
28px multiples"). Our `agentic` slice already applies it (`agentic.py:275`) and
lands at ≤380 tokens. Our `doc` slice does not, and lands at up to 16,598.

## 2. What the collapse looks like numerically

- **When:** "Around 900 steps into this process (one epoch is 1035 steps), we
  observed a **sharp drop in the loss**" → step 900 = 87% of epoch 1 =
  **57,600 samples** (batch 64 × 900).
- **Shape:** a long plateau, then a sharp, **sustained** drop — the projector
  "figures out how to align the image features with GLM's latent space".
  Harry's article gives no loss values, and 0xSero's repo publishes no curve
  (11 files, no logs), so the magnitude is not documented anywhere.
- **Operational detector (0xSero, `train_sft.py::_check_grok`):**
  `"if drop > 0.5:  # >0.5 nats drop in one log interval = grok signal"`,
  evaluated every `log_every=10` steps, from step 800.

**Our log, measured with the reference's own threshold** (`scripts/analyze_probe.py`):

| statistic | value | verdict |
|---|---|---|
| max single-step loss drop | +2.504 nats (step 2,524) | "GROK" |
| max 10-step-window loss drop | +5.273 nats (ending step 1,475) | "GROK" |
| 10-step windows dropping > 0.5 nats | **271 of 3,990** | "GROK" 271× |

**So the reference's threshold is useless on a per-step log of a heterogeneous
batch.** It fires on noise: our loss is a token-mean over a batch whose
composition (three manifest groups, a wide `n_vis` range, a decaying LR) makes
single-step moves of 1–2.5 nats ordinary. A collapse is a *sustained level
change*, not a single-window drop.

**Our run's level** (the real test): the EMA oscillates in **[1.06, 1.50]** from
step 800 to step 4,000 and ends at **1.08**. The plateau never breaks. There
was **no collapse** — by any reading of the reference's description.

## 3. Can a batch-16 / 2B-backbone probe reach it?

**Sample count: yes.** 57,600 samples = step 3,600 in our run (we ran to 4,000).
So we *did* pass the reference's grok sample count — our step 900 was only 25%
of it, but the run as a whole reached 111% of it. The plateau at 57,600 samples
is therefore a real absence, not an early stop.

**Three differences that plausibly suppress the event**, in the order they are
worth testing:

1. **The LR decayed to nothing before the sample count where the reference
   grokked.** Ours: warmup 100 + cosine to 0.10×peak over 4,000 steps → at
   step 3,600 (the reference's 57,600 samples) our LR was **≈0.11 × 5e-4 =
   5.5e-5**; the reference held a **constant 5e-4** straight through its
   grok. A sharp alignment event needs a live LR to happen. *Cheapest test in
   the project: one config line (constant LR, horizon = the run length).*
2. **Our visual-token regime was 3–40× wider.** The reference's images are all
   ≤383 tokens; ours were 16 to 16,598 (mean 685), 43.9% above 512, 16.9%
   above 1,024, and **1.7% above the 4,096 context limit** — those rows were
   *truncated* (head kept), so the projector was asked to align features whose
   spatial extent had been cut, with the answer crushed to ≤2 tokens. That is
   not a harder version of the reference's task; it is a corrupted one.
3. **The alignment task itself is different.** Harry's 66k images with "simple
   question/answer pairs" are mostly *object naming* — a coarse, low-detail
   mapping. Our data is UI grounding and chart/document reading, where the
   answer depends on fine detail spread across a large image. **After capping
   at 383 tokens, a 3.9 Mpx chart page becomes a 300k-px thumbnail whose
   12-px glyphs are 3-px smears — the doc slice's answers become unreadable.**
   *This is a hypothesis, not a source claim, but it is testable: it says the
   reference's operating point transfers to the agentic slice and does NOT
   transfer to the doc slice.*

## 4. What this implies for the plan

- **The cap is per-slice, not global** (D6): agentic **383** (the reference's
  own regime), doc/OCR **1,024–2,048** (dense enough to read a chart, and
  still inside the 4,096 context with room for the text). A global 383 would
  quietly destroy the slice we added.
- **The next probe should be the reference's recipe, not ours:** constant LR,
  and the agentic slice (where the regime matches). If the grok appears there,
  the recipe is confirmed and the doc slice becomes a per-slice-budget project
  rather than a blocker.
- **The plateau of the 4,000-step probe is not evidence against the recipe.**
  It was produced with a decaying LR, a 40× wider token regime, 1.7% truncated
  rows, 39% of the questions unserved, and the wrong geometry (M30). Five
  independent defects, all now fixed or measured.

## 5. Reproducing the numbers in this document

```bash
python scripts/analyze_probe.py --log data/probe_log.jsonl   # the split, the slice table, the grok test
```

The grok test in the script uses the *sustained* criterion (the EMA level over
the plateau) and ALSO reports the reference's naive per-window threshold, with
the caveat above.
