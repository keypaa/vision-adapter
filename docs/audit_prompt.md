# Audit prompt — Vision-Adapter, fresh session

Copy everything below into a new session with no prior context.

---

You are auditing a vision→LLM adapter trainer at
`/home/keypaa/Projects/DSV4-0731/Vision-Adapter`. It trains a small projector
that maps frozen MoonViT-V2 image embeddings into a Qwen3.5-2B backbone so the
model can do computer-use (UI screenshot → next action).

**A 20-hour training run once completed with a falling loss curve and taught
the model nothing at generation time.** That is the bar for this audit: a bug
that lets a run look healthy while producing a useless adapter is the failure
mode we are hunting. Assume it is still in the code.

## The hard part, and what makes this audit different

Most bugs of this class are invisible to a unit test. On 2026-09-30 a suite of
231 green tests coexisted with three real bugs, each of which only surfaced on
a real GPU run:

- `steps = max_steps or 5` made `max_steps=None` a silent 5-step run that
  exited 0 — indistinguishable from success in the logs.
- fp16 auto-selected on a T4 NaN'd all 6 projector params in the backward,
  while the forward looked perfectly healthy.
- The measured-geometry feature was fully wired but defaulted to off, so
  every run silently used the old approximation.

So: **do not treat a passing suite as evidence.** Where you can, reproduce on
real data or a real GPU. Say explicitly when you could not execute something
and what that leaves unverified.

## Discipline

1. **Verify each claim against the current tree before acting on it.** A
   review written against a stale tree will name risks that are already
   closed. If a claim turns out to be false, say so plainly and move on — do
   not manufacture work.
2. **Prefer measurement to argument.** A 200-row sample that settles a
   question beats a paragraph of reasoning about it.
3. **Separate "broken", "wrong by design", and "unknown".** Some divergence is
   intentional. Flagging a deliberate choice as a bug wastes the time you are
   trying to save.
4. **Quote the code.** `file.py:123` plus the actual line. Not a paraphrase.
5. **Do not fix anything.** Report only. Rank by whether it can produce a
   healthy-looking run with a useless adapter.

## Context you should have

Work done 2026-09-30, all verified on live data. Assume these are correct and
do not re-derive them — they are here so you do not flag them as issues:

- **Geometry.** The synthetic `grid_for_nvis` stand-in inverted orientation
  (differed from the true grid on 38.1% of rows, swapped portrait/landscape
  on 18.3%, aspect error to x89). The true grid is a deterministic function of
  image dims under the preprocess contract, recovered and written per-row as
  `grid_thw` into `train_manifest_grids.jsonl` — 117,600/117,600 verified
  against the `n_vis` in the embedding key index. Rows without a grid fall
  back to synthetic, and `geometry_guard()` refuses to start without
  `--allow-synthetic`. `grid_source` in the run header records the regime.
  Note `n_vis` alone is NOT sufficient: n_vis=364 covers two distinct real
  grids.
- **Validation.** The streaming path had none. It now probes every
  `cfg.val_every` steps (default 250) on `train_manifest_val_disjoint.jsonl`
  (1,272 rows, 0 emb overlap with train, 84.6% agentic), materialized once to
  a ~10 MB local cache. The loss reuses `train_step_qwen`'s exact recipe so a
  val/train gap is real overfitting.
- **fp16.** `resolve_dtype()` never returns fp16 for `auto`: bf16 for cc>=70
  (T4 included), fp32 below. Measured: fp16 forward fine (hidden absmax 48,
  loss 9.36) but backward NaN'd 6/6 projector params; bf16 gives gnorm 522,
  0/6. A GradScaler does not help — the overflow values are activations, not
  fp16 parameter grads.
- **Collate.** `L <= max_len` is capped, and `strip_text_image_markers` drops
  the `|begin_of_image|…|end_of_image|` block (265 Qwen tokens on 45% of
  rows) before tokenization.
- **`precompute`.** Used to validate args and return None without generating
  anything. Now delegates to the real MoonViT pipeline, and refuses
  non-local backends loudly.
- **`max_steps=None`** means unbounded, not 5. An unbounded run must pass
  `lr_horizon` because the cosine LR needs an end.

## What we know is open — do not treat as new findings

- The **legacy `[BOS][vis][user][answer][EOS]` layout is still the default.**
  Native Qwen framing (`[vision_start][image_pad×N][vision_end]` +
  `mm_token_type_ids` + native mRoPE) is opt-in via
  `VISION_ADAPTER_NATIVE_TRAIN=1` and **has never run in a real training**.
  The user's position: a model that learns computer use has learned to see, so
  the 45/45/10 dataset mix is accepted as-is.
- The user wants **per-model injection modules** rather than one generic path
  with conditionals in it. The divergence is real and already visible: Qwen
  forbids `input_ids` + `inputs_embeds` together (hence `embeds_for`);
  DeepSeek-V4 hash-MoE gates on `tid2eid[input_ids]` (hence the
  `visual_inject` hook). This is a design direction, not a bug.

## The real question

**Qwen3.5-2B trained for 20 hours with a falling loss and learned nothing at
generation.** Assume the curve was real and the model still could not do the
task. Work backwards from that.

Candidate explanations, none confirmed. Rank them and say which measurement
would separate them:

- The sequence layout at training time does not match what the model sees at
  multimodal inference, so it learned a format it never reuses.
- The loss is dominated by easy tokens (the answer is 3-12 tokens for
  agentic) and the projector converged to predicting them while ignoring the
  image. **Check whether the projector actually depends on the visual
  embeddings at all** — e.g. is the gradient reaching the projector through
  the visual path, or only through text?
- Supervised signal too sparse or too templated to teach grounding.
- The learning-rate schedule, warmup, or batch composition never leaves the
  plateau it appears to enter.
- Something in the injection silently degrades the image representation
  (dtype, normalisation, scale) without breaking the loss.
- The task as supervised (next low-level UI action) is not the task
  intended (screenshot comprehension).

Also worth checking, in rough priority order:

1. Does the projector receive gradient from the visual path, and is the
   projector actually changing? A run that saves an untrained-equivalent
   projector would look fine in the loss.
2. What does the projector output look like relative to the embedding
   distribution it must match? `ScaledHourglassProjector` exists because raw
   head outputs were measured at rms 16.42 against a 0.0131 table — verify
   that ratio still holds on the current data and that the fix is active.
3. Is `target_rms=0.02` right for the current backbone, or is it inherited
   from a measurement on a different setup?
4. Generation path vs training path: does eval use the same injection, the
   same positions, the same prompt framing as training?
5. Is anything in the resume/checkpoint path able to restore a stale
   projector, so a run appears to continue when it restarted from scratch?
6. Token-level: how many supervised tokens per example, and what fraction of
   the gradient signal comes from the answer vs the image region?

## Deliverable

A ranked list. For each finding:

- **What** — the defect, at `file.py:line`
- **Why it matters for the failing run** — the causal path to "healthy loss,
  useless adapter", or an explicit statement that it does not connect
- **Evidence** — measured, quoted, or reasoned; label which
- **Severity** — can it produce a healthy-looking useless run? can it waste
  GPU time? is it cosmetic?

End with:

- The single measurement you would run first, and what result would change the
  diagnosis
- What you could not verify without a GPU or without running training, stated
  plainly
- Anything you believe is fine, so we do not spend time on it again

Be blunt. "I could not determine this" is more useful than a confident guess,
and far more useful than a restatement of what the code appears to do.
