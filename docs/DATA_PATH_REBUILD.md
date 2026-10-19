# DATA PATH REBUILD — evidence, decisions, and prioritized work

Status: **PLAN** (nothing implemented) · Last updated: 2026-10-19
Scope: the corpus (what bytes exist, at what resolution) and the loader (how they
reach the GPU), for both the Qwen3.5-2B probe and the DeepSeek-V4-Flash hero run.
Related: `docs/DATA.md` (corpus facts) · `docs/NEXT_STEPS.md` (project roadmap —
this doc is the detailed plan for the data path) · `docs/audit_datapath_prompt.md`
(earlier audit, symptoms only).

---

## 0. Goal

> **The GPU never waits on data.** Wall-clock == GPU time, at hero scale, on any
> box we control, with one data format and one plan.

Secondary: the probe must be able to run *repeatedly* (a run is an experiment;
the recipe needs many of them), so the data path must not cost more wall-clock
than the science.

---

## 1. What we measured (all from the run's own artifacts)

Source artifacts, external, not copied here (steward rule 5):
`keypa/vision-adapter-probe-checkpoints` → `probe_log.jsonl` (4,000 train
records), `train_4000.log` (14,606 console lines), `train_manifest.jsonl`
(sha256 = the run's recorded `manifest_sha256`).

| # | Measurement | Value |
|---|---|---|
| M1 | Probe wall clock (4,000 steps, resumed run = steps 401–4000) | 24,366 s (406.1 min) |
| M2 | GPU compute (`Σ step_ms`) | 14,673 s — **60.2%** of wall |
| M3 | Data-path gap (wall − `step_ms`) | 7,166 s — **29.4%** of wall |
| M4 | …of which main-thread **network** (console `streamed … in Ns`) | 5,924 s — 83% of the gap |
| M5 | …of which local (decode + collate + log + ckpt) | 1,242 s — 5.1% of wall |
| M6 | Startup (resume skip + index build 32 s + plan) | 2,527 s — 10.4% of wall |
| M7 | Bytes over the wire / unique / useful | 957.3 GiB / 798.0 GiB / 250.4 GiB → **3.19× overfetch** |
| M8 | Bytes fetched **twice** (main + background path, 359 shard+rg pairs) | 159.0 GiB — 42% of main-thread fetch time |
| M9 | Effective Range rate, by span size | <100 MiB → 7.4 MiB/s · 100–200 → 36 · 200–500 → 55 · 0.5–1 GiB → 59 · 1–2 GiB → 96 · >2 GiB → **171** |
| M10 | Steps with gap > 5 s | 462 of 3,600 (5,356 s) |
| M11 | Resume cost (step 401) | 22.7 min — re-fetched ~190 row groups to skip 6,400 rows |
| M12 | Steps with L > 4,096 (padded batch length) | **117** of 3,600 (3.2%) — mean 56.4 s/step vs 2.32 s → **45% of all GPU time** |
| M13 | Loss on those 117 steps vs the rest | mean **3.27** (median 2.97) vs 1.17 (median 1.09) |
| M14 | Loss curve | falls fast for 200 steps, then **plateaus ≈1.1 for 3,800 steps — no grok collapse** |
| M15 | Manifest rows with a duplicate `emb` key | 45,393 of 117,600 (**39%**) — 8,653 images carry 2–6 *different* questions |
| M16 | Rows actually served (loader collapses duplicates) | 71,051 unique positions |
| M17 | `n_vis` by group (from the key index) | `agentic` median 364 / **max 380** · `doc` median 960 / mean 1,093 / p99 5,184 / **max 16,598** · `conv` median 368 |
| M18 | Share of rows above candidate caps | >512: 43.9% · >1,024: 16.9% · >2,048: 2.3% · >4,096: 1.7% |
| M19 | Byte concentration of the tail | >4,096 → 1.7% of rows = **12.8% of bytes** (77.7 GiB) · >1,024 → 16.9% of rows = 38.7% of bytes |
| M20 | Corpus size at each cap — three different numbers, do not mix them | **manifest rows** (117,600, with duplicates): 609 GiB now → 312 GiB @383 · 363 @512 · 527 @1,024. **served positions** (71,051, dedup'd): **291 GiB now → 182 GiB @383**. **all images embedded** (138,987, "do all"): ~394 GiB @383 |
| M21 | `process_image` cost, same 12.8 Mpx image: **OLD geometry vs 383 cap** | already-resized image 11 ms vs full-res 461 ms (43×); fp32 canvas 3 MiB vs 147 MiB. **But** the real precompute must *do* the resize: patching the real `MAX_PATCHES` constant gives 419 ms → 96 ms (**4.4×**, the resize itself is ~85 ms) — `scripts/bench_precompute.py --synthetic-old` measures both on the target host |
| M22 | Current precompute CPU/GPU balance (sandbox, stubbed ViT) | CPU-side ≈ 30 ms/img vs GPU-side 0.6–8 ms/img → **the CPU is the bottleneck by 3.6–48×**; decode 9.4 ms, preprocess 18.6 ms, per-image `torch.save` 1.7 ms |
| M23 | **Images repo, per-subset encoded size** (read via Range on the `size` column only, ~100 KB/shard) | 138,987 images, **33.80 GiB** (mean 255 KiB). Full-res-looking: `websight` 1,096 KiB mean / 11,160 KiB max · `textvqa` 949 / 25,078 · `docvqa` 932 / 11,625 · `infographic_vqa` 815 / 3,866 · `screen2words` 500 / 3,663 · `vqav2` 468 · `visual7w` 325 · `okvqa`/`aokvqa` 404 · `ocrvqa` 248 · `waveui` 196 (lossless PNG of a capped image). Small: `miniwob` 7.5 · `aitw` 39 · `mind2web` 52 · `guiact-web-multi` 56 · `plotqa` 29 · `chartqa` 39. **93% of the bytes are in subsets >120 KiB/img** |
| M24 | **Served vs manifest composition** (the loader collapses duplicate `emb` rows) | agentic 52,924 manifest → **52,924 served (73.3%)** · doc 52,908 → **12,780 (17.7%)** · conv 11,768 → 6,503 (9.0%). The doc slice asks **up to 54 questions per image** (3,983 images with ≥6 questions, all chart/figure QA: *"What is the difference between the Balance of Payment (as % of commercial service exports)"* → answers `No.`/`Yes.`/`0.`/`2.`) |
| M25 | **Collate budget pathology A** — `core.py:305-318` | for `n_vis ≥ 4,090`: `budget_text = max_len - nv - 2 ≤ 2` → **the answer is cut to ≤2 tokens and the question to 1**. **1,986 rows (1.69%)**, all `doc`, carrying **27.7% of all answer words**. Present at HEAD *and* in the 4,000-step run's revision |
| M26 | **Collate budget pathology B** — long text squeezes the image | `excess` trimming removes the *visual* span when text is long → the model sees a cropped image. The 8 worst-loss steps (loss 5.4–7.0) all have `n_vis ≈ 350-414` and **zero** rows with `n_vis > 4,090`: their length comes from long *questions*, and the visual span is what gets cut |
| M27 | **Loss by group / by image size** (replay + log join) | flat everywhere: agentic 1.236 · doc 1.225 · conv 1.235; n_vis 0-100 → 1.239, 101-500 → 1.237, 501-1000 → 1.208, 1001-2000 → 1.193, >4,090 → 1.278. **The plateau is not a "big images fail" effect** — it is uniform |

Replay validation: shard order, row-group geometry (1,088/1,109 console fetch
sizes match), 92 shards opened, 1,998 row groups and 798.0 GiB unique all
reproduce the console log exactly.

---

## 2. Root cause — and what it is *not*

**It is not the row group.** A row group is parquet's internal chunk (64 rows
here, from `pack.py:pack_rows(batch_size=64)`); it only became the unit of
*transfer* because we chose to range-fetch parquet remotely. It is a symptom.

**It is not the Modal-volume deletion.** That was a correct cost decision
($0.85/day for 930 GiB). It forced the corpus to HF and the trainer to stream
instead of materialize — a retrofit, not a mistake.

**It is: the doc slice was precomputed at ~40× the reference pixel budget.**

The reference recipe (0xSero's builder for the mix we ingested; Harry
Partridge's method) resizes images to **≤300,000 px**. With our tokenization
(14 px patch, 2×2 merge → 28 px stride — the *same* stride as the reference)
that is **≤383 visual tokens**.

- `agentic` slice: rule applied (`vision_adapter/data/agentic.py:275`) →
  median 364, **max 380** tokens. Exactly the reference regime. (M17)
- `doc` slice (the Cauldron half): **no pixel cap anywhere** —
  `cauldron.py` has none, and `navit_resize` allows `MAX_PATCHES = 65,536`
  (12.8 M px → 16,384 merged tokens). Median 960, max 16,598. (M17)

One divergence, four consequences:

| consequence | mechanism | evidence |
|---|---|---|
| 45% of GPU time in 3.9% of steps | doc rows are 3–40× the reference token count → batch cost = longest row (padded L) | M12 |
| Those steps carry 3× the loss | the long rows are the hard ones (dense docs, infographics); at HEAD they are also **truncated** to `max_seq_len` (head kept), so the answer region can be cut → unlearnable | M13, `core.py:308,314` |
| 830 GB corpus, 3 GiB row groups, 29% of wall on I/O | the same rows are 3–40× bigger than they need to be | M7, M9 |
| no grok collapse | the projector (a *per-token* MLP) now has to align over a 1000× token-count range, and 1.7% of rows are truncated | M14, M18 |

**Corollary: fix the payload and the transport problem largely disappears.**
At the reference budget the corpus is ~182 GiB (served, dedup'd) or ~394 GiB
(all images embedded) — it fits on the 500 GB rental box, whole-file download is
minutes, and range-fetching becomes a first-class remote path rather than the
only path.

### 2.1 State of the two HF repos (checked 2026-10-19, metadata only)

> **CORRECTION 2026-10-19 (later):** an earlier version of this section claimed
> the images repo was already capped. **It is not.** The table below is the
> corrected state; the per-subset map is M23.

| repo | state | evidence |
|---|---|---|
| `keypa/vision-adapter-images` | **NOT capped** — 33.80 GiB of image bytes for 138,987 images (mean 255 KiB/img). The README's "Each image has been resized to ≤300k pixels" is **false for the cauldron rows**; `dataset_info.json`'s `num_bytes` (6.76 GiB) is stale. The **agentic** slice *is* capped (its n_vis ≤ 380, M17); the **cauldron** slice is not (M23) | HF API tree (33.40 GiB in 17 files) + per-row `size` column read via Range (M23) |
| `keypa/vision-adapter-embeddings` | still the old geometry — 103 shards, **883.8 GiB**, `emb_0099` = 60.2 GB | HF API tree (file sizes) |
| `keypa/vision-adapter-manifests` | unchanged since the run | — |

→ The embeddings are consistent with the *uncapped* images: the doc slice's
16,598-token embeddings came from 12.8 Mpx PNGs that are still in the images
repo. **No upstream re-fetch is needed** (the images are already here, 33.8 GiB),
but they *do* need capping.

**The fix is one constant.** `vision_adapter/models/preprocess.py:14`
`MAX_PATCHES = 65_536` (12.8 M px) is the only thing standing between the
reference budget (≤300k px ≈ 1,531 patches → ≤383 merged tokens) and what we
embed. `core.grid_from_dims` imports `_resize_size` from the same module
(`core.py:119`), so the geometry contract stays single-sourced — but the
manifest's stored `grid_thw` must be **regenerated** after the change (W9b).

The pixel cap must still be enforced in the *dataset* stage code (W6) so a
future rebuild cannot regress — but it is a reproducibility guard now, not a
prerequisite.

---

### 2.2 Scope change 2026-10-19: **Colab is dropped**

Decision: Colab T4 is no longer a target (too slow, weak GPU). Consequences,
all of which simplify the work:

| was driven by Colab | now |
|---|---|
| 15 GB memory ceiling → the byte-capped-RG repack, the `MAX_RG_ROWS` guard, the `3×RG + 5.4 GiB` peak-memory analysis | **moot** — rental boxes and Modal have real RAM; with a 383 cap a row is 2.9 MiB |
| 12 GiB `rg_cache` cap (chosen for a Colab disk) | **moot** — the cache becomes an opportunistic, disk-sized LRU (W15) |
| "must work on 15 GB *and* on a large box without a branch" | **"must work on a rental box (persistent disk) and on Modal (no volume) without a branch"** — still a branch, but now it is *disk policy*, not a memory ceiling |
| Colab as the smoke-test host | smoke tests run on the rental box; the 2-step "fetched 3.6 GB" symptom becomes a row-granular-plan issue, not a memory issue |

What does **not** change: streaming stays first class (Modal has no volume),
the corpus format and the plan are host-independent, and the resume path is
unchanged.

### 2.3 What the adversarial audit changed (2026-10-19)

A hostile review of this plan returned three findings that survive verification
and one that does not. All of them change the *order* of the work.

**Accepted (verified locally):**

1. **The collate's text/visual budget is broken in two ways** (M25, M26). The
   overflow guard protects the answer by *starving* it: for `n_vis ≥ 4,090` the
   answer becomes ≤2 tokens and the question 1 token (1,986 rows, 27.7% of all
   answer words); for long *questions* the visual span is squeezed instead and
   the model sees a cropped image (the 8 worst-loss steps, loss 5.4–7.0, are all
   `n_vis ≈ 350-414`). The docstring's "answer has PRIORITY on the text budget"
   is true only for the answer, and the question and the image are collateral.
   **This is a code bug, not a corpus problem, and it is fixed in a day.**
2. **The served mix is not the manifest mix** (M24): `doc` is 45% of the manifest
   but **17.7% of what is actually served** (12,780 images, up to 54 questions
   each, mostly chart/figure QA). Every "45% doc slice" statement in this plan
   was a manifest-level number.
3. **The rebuild was scheduled before the question it answers.** The cap is a
   *hypothesis* ("oversized doc images are why we do not grok"), and the
   collate bug + the unserved 39% of questions are *known defects* that
   invalidate the experiment either way. Fixing the known defects first costs
   ~1 day of code and ~5 GPU-hours, and its outcome **cancels or confirms** the
   2–4 week rebuild.

**Rejected (with reasons):**

- *"The 383 cap destroys the doc slice."* The cap would indeed make a 3.9 Mpx
  docvqa page unreadable (a 0.28× linear downscale turns 12-px glyphs into
  3-px smears). But that is an argument for a **per-slice budget** (UI 383,
  OCR/chart 1,024–2,048), not for abandoning the cap — and the *current*
  behaviour (16,598 tokens, truncated answers, cropped images) is worse than
  either. The cap's open question is its *value*, not whether a budget should
  exist.
- *"The per-token MLP cannot be the mechanism."* Right that the projector is
  per-token, so a wider token-count range does not change the mapping. The
  mechanism is elsewhere and still real: the *frozen LM* must attend over
  16k visual positions, and the batch's padded length (L) is set by its longest
  row (M12: 117 steps, 45% of GPU time). M27 also shows the loss is *flat*
  across image sizes, so the size effect is a **cost** problem, not a
  **quality** problem.

**Net effect on the plan:** the order becomes *fix the known defects → re-run
the probe on the current corpus → then decide the cap and the rebuild from the
result*, with the rebuild demoted from P1 to conditional.

## 3. Decisions required

### D1 — Visual-token budget (the cap). *Do not take this on faith.*

What we **know**:
- 0xSero's dataset card: images are regenerated with "resize to ≤300k pixels,
  dimensions rounded to 28px multiples". With our stride that is ≤383 tokens.
- Our own `agentic` slice already does this and lands at ≤380 tokens.
- 0xSero's `fable-glm-vision` is a **working reproduction of the recipe on this
  mix** → the ~383-token operating point is *validated* for the recipe.

What we **do not know**:
- Harry's own data pipeline (his post is on X, not published) → "Harry capped at
  383" is an inference through 0xSero's builder, not a direct citation.
- Whether a *larger* budget (more detail) would be better. The reference never
  tested it, so it is untested, not proven worse.

Independent of the reference, our own hard constraints:
- `max_seq_len = 4096` (TrainConfig) → 1,986 rows (1.7%, **12.8% of bytes**) are
  truncated today; truncated examples are unlearnable noise.
- Batch cost is set by the **longest row** (padded L) → the tail sets step cost
  for the whole batch.

**Options**

| cap | corpus | rows above cap | batches | notes |
|---|---|---|---|---|
| **383** | 312 GiB | 46.7% (downscaled) | uniform, ≤ ~500 tok | the reference operating point; same as our agentic slice; **recommended default** |
| 1,024 | 527 GiB | 16.9% | ≤ ~1.1k tok | 3× reference cost/step, no validation |
| 4,096 (context) | 593 GiB | 1.7% (still truncated) | L up to 4k | status quo minus the worst |

**DECIDED 2026-10-19: cap = 383.** It is the only value with (a) a validated
provenance, (b) internal consistency with the agentic slice, and (c) uniform
batches. A cap *downscales* images (`agentic.py` already does exactly this);
it does **not** drop rows. The bounded A/B (W3) is retained as an *optional*
sanity check, not a gate — 383 is the reference operating point and the choice
of a target, not a hypothesis.

Scope of the change: rows with `n_vis > 383` = 46.7% of the manifest (the doc
slice + 12% of conv); the `agentic` slice is unchanged.

### D2 — Corpus format: can we avoid row groups? **Yes.**

Row groups only exist because we distribute as parquet *and* fetch byte ranges.
Options:

| format | local read | remote read | notes |
|---|---|---|---|
| parquet (today) | `read_row_group` materializes a whole RG; `iter_batches` avoids it | Range per RG → **3.19× overfetch** (M7) | RG granularity is the waste source |
| **flat blob + per-row index** (recommended) | `np.memmap` + offset → exact row, zero copy | byte range per row using the index → **1.0× overfetch** | one file per shard; index = (offset, length, n_vis) per row ≈ 30 B/row ≈ **4 MB total** |
| per-row files | trivial | 137k HTTP objects — worse | ✗ |
| WebDataset/tar | sequential only | sequential only | fine for "read everything once", bad for a partial plan |

Recommendation: **store the payload as a flat concatenated bf16 blob per shard +
an explicit per-row index**; keep parquet only for the *metadata* columns
(key, n_vis) if convenient. The "unit of transfer" becomes one row (~2.9 MiB
at the 383 cap) instead of one row group (up to 3 GiB) — and it becomes a
*choice* rather than a parquet implementation detail.

**Concrete layout (decided 2026-10-19):**

```
emb_XXXX.bin   rows concatenated, each n_vis×4096 bf16, in shard row order
emb_XXXX.idx   n_vis per row (uint16 × 1360 = 2.7 KB per shard)
key_map        key → (shard, row) for the manifest join (~3–6 MB total)
```

Because the blob is a plain prefix sum of `n_vis`, the offset of row *r* is
`prefix[r] × 8192` — the **whole per-shard index is ~278 KB** for the corpus
(137k × 2 B), and the key map is the only multi-MB artifact. A plan is just a
list of `(shard, row)`.

**Granularity without dead time (the concern in the review):** the transfer
unit is *not* a row. The trainer asks for the next batch (16 rows); those rows
are physically adjacent in the plan, so their byte ranges are adjacent and
**coalesce into one Range request**. Prefetch depth 2–3 requests the *next
few batches* as one aligned range. So: exact bytes (1.0× overfetch), request
count unchanged versus today, no dead time.

**The invariant that matters:** bytes fetched = bytes of the planned rows, so
**fetch time ∝ plan size** — 100%, 50%, 10% or 1% of the corpus all cost
proportionally. Today it does not: row-group granularity adds a 3.19× constant
(M7) and `build_epoch_plan` is *shard*-granular, so a 300-row plan still pulls
whole shards. Making the plan **row-granular** is what buys the invariant (and
satisfies the original "no download of data we will not train on").

### D3 — One format, one plan, two row-sources

"Local" = a box we control with a **persistent disk** (Molab / RunPod /
LightningAI) — we can install `hf_transfer`, choose the cache dir, and the
cache survives between runs. "Remote" = **Modal** (ephemeral containers, no
volume, priced storage we would rather not buy). *Colab is out of scope
(§2.2).*

Requirement: **the same corpus, the same index, the same plan, the same row
order** in both — only the transport differs, behind one interface:

```
RowSource (interface)
  ├─ LocalDiskSource   : materialize shards (whole objects, hf_transfer) → memmap + index
  └─ RemoteRangeSource : per-row byte ranges from the same index (bounded cache)
```

Both yield identical rows in identical order → the plan hash and resume
semantics are backend-independent. The branch lives in **one** place, not in the
loader (this is the "no code branch per machine" requirement, done properly).

Note on Modal economics: a 930 GiB volume cost $0.85/day; a ~180–300 GiB corpus
volume is ≈ $0.16–0.27/day — i.e. with the payload fixed, "keep it on a Volume
and read locally" becomes affordable on Modal too, and the two backends may end
up being the same backend with a cold-start variant. *(The "no volume" rule
still stands — D3.)*

**DECIDED 2026-10-19:**
- **No Modal volume.** Streaming is first class and stays first class.
- **Disk is an opportunistic cache, not a store** — bounded by an explicit
  limit, on every host; on Modal that limit is the container's ephemeral disk.
  Nothing may *depend* on the cache persisting.
- **Backend independence is a hard requirement**: on Modal vs Molab vs RunPod vs
  LightningAI, the *same* corpus, *same* index, *same* plan, *same* row order,
  *same* results. Only the transport differs, behind one `RowSource` interface.
  The host must not change the science.

### D4 — Probe recipe alignment (the grok question)

The reference conditions we are not running: batch 64 (we use 16) and a
**different backbone** (the reference grokked on a 304B+ MoE; the probe runs on
a 2B model — 150× smaller, never mentioned in the earlier version of this
plan). Our LR cosine-decays over 4,000 steps, so by the reference's grok point
(~step 900) it is already at ~0.14·peak, which is *not* obviously fatal — the
reference's own schedule is unpublished, so a constant-LR arm is a one-line
experiment. Our answers are short **by word count** (97.2% ≤5 words, mean 2.0)
but that is not the same as "a few *tokens*": the token-weighted mean is ≈10
tokens, so the "short answers" requirement is *not yet validated* — worth
measuring with the real tokenizer.

Two further confounds that are **not** the pixel budget: the 39% of questions
never served (M15/M24) and the collate budget pathologies (M25/M26).

### D5 — Duplicate-question policy

Multi-question-per-image rows are **intended** (M15, M24): 8,653 images carry
2–54 different questions, same `g` (mostly chart/figure QA — up to 8 questions
per figure). The loader collapses them (`stream.py:793-794`,
`want[rg][pos] = r`) → 45,393 rows (39%) silently unserved, while `samples_seen`
counts them as served. Fix = key the plan rows, not the position. **This changes
the served stream → new plan version → checkpoints from the old plan cannot be
resumed** (they must keep pointing at the old corpus, which we keep).

### D6 — The visual-token budget (was D1's value; now a per-slice question)

383 is the *reference* operating point and the right default for the **agentic**
slice (which already sits at ≤380). For the **doc** slice it is a policy choice
with real downside: a 3.9 Mpx docvqa page downscaled to 300k px is a 0.28×
linear reduction — 12-px glyphs become 3-px smears, one patch per ~5 characters.
So the budget should be **per slice** (UI 383, OCR/chart 1,024–2,048) — and
*which* value is an empirical question, to be settled by the P1 probe re-runs
(M27 shows the loss is currently flat across sizes, so the size axis is
untested, not dismissed).

---

## 4. Prioritized work

Cost units: **S** ≤ half a day · **M** 1–3 days · **L** > 3 days (or GPU-hours).
"Resume risk": can a checkpoint from the current corpus still be resumed?

### P0 — Fix the known defects first (no corpus rebuild)

> **Reordered 2026-10-19 (§2.3).** These three items are *known bugs* that
> invalidate any experiment run before them are fixed, and they cost ~1 day of
> code and no corpus work. The rebuild (P1) is demoted to **conditional**: the
> probe re-run in P0 decides whether it is needed at all.

| id | item | why | cost | resume risk |
|---|---|---|---|---|
| **W0** | **Fix the collate text/visual budget** (`vision_adapter/core.py:305-318`): reserve the answer first, then the question, then give the image whatever is left; if the image would fall below a usable minimum, drop the row instead of cropping it. Today: `n_vis ≥ 4,090` → answer ≤2 tokens + question 1 token (1,986 rows, 27.7% of all answer words, M25); long questions → the *image* is cropped (the 8 worst-loss steps, 5.4–7.0, M26) | M25, M26, M27 | **S** (~1 day) | none (same rows/order; the *labels* change, so a resumed run is statistically-equivalent-not-identical, same as the existing contract) |
| **W0b** | **Serve the duplicate questions** (D5): the plan must yield every question of an image, not one per position. Today 45,393 rows (39%) are never served | M15, M24 | **S** | **yes** (served stream changes → new plan version) |
| **W0c** | Drop the **long-answer rows** (W1b, 645 rows) and log **per-group loss** (agentic / doc / conv) + `wait_ms` so the next probe says *which slice* learns and *where* the wall goes | source recipe, D4, M27 | S | **yes** (manifest sha) |
| W1 | ~~visual-token cap = 383~~ → moved to **D6 (per-slice budget)**, decided *after* the P0 probe | M17–M20 | S | none |
| W2 | Choose the corpus format (D2) + the two row-sources (D3) | M7, M9 | S | none |
| W4 | Probe recipe arms (D4): (a) current LR schedule vs constant LR; (b) batch 16 vs 64 if it fits; (c) the backbone confound — the probe is 2B, the reference 304B+ | M14, D4 | S | new runs only |
| W5 | Duplicate-question policy (D5) → folded into W0b | M15 | S | — |

### P0.5 — Re-run the probe on the CURRENT corpus (≈5 GPU-hours, 0 rebuild)

| id | item | gate |
|---|---|---|
| **W17a** | Probe re-run #1: W0 + W0b + W0c applied, current corpus, current recipe. ~4,000 steps, `wait_ms` + per-group loss | **the grok cliff appears** → the pixel cap was never the problem → **P1 is cancelled** (saves 2–4 weeks) and the remaining work is the data path alone. **no cliff** → the plateau is not the data's fault; the search moves to the recipe (W4 arms) and the cap becomes a *cost* decision, not a correctness one |
| **W17b** | Probe re-run #2: the winning recipe arm from W4 (constant LR or batch 64) | a cliff, or a documented statement that none occurred |

### P1 — Rebuild the corpus (**conditional on W17a**)

| id | item | why | cost | resume risk |
|---|---|---|---|---|
| W6 | **Cap the precompute pixel budget — one constant**: `vision_adapter/models/preprocess.py:14` `MAX_PATCHES` 65,536 → the reference budget (≤300k px ≈ 1,531 patches). Applies to the cauldron rows at precompute time; the agentic rows are already under it and are unaffected. `core.grid_from_dims` imports `_resize_size` (`core.py:119`) so the geometry contract stays single-sourced | M17, M23, §2.1 | S | none |
| W7 | **Re-precompute the whole corpus from the images repo** (input 33.8 GiB, 138,987 images) and **write the corpus blob + index directly** — no 137k `.pt` files, no separate pack pass. The images are already in the repo (no upstream re-fetch); the cap is applied by W6 | M17, §2.1 | L (GPU-hours, but see W7a) | none (new corpus) |
| W7a | **Overlap CPU decode with GPU forward** in the precompute (producer thread pool → patch batch queue → ViT → blob writer). The current loop decodes/resizes every image *serially* while the GPU idles (`precompute_colab.py:136-143`) | the 6 h run was largely CPU-serial decode + per-image `torch.save` | M | none |
| W7b | **Write the corpus as blob+index in shard order** (one `.bin` + one `.idx` per shard, atomic rename), so W8 disappears and the output is directly consumable by both row-sources | D2 | M | none |
| W7c | **Resumability at shard granularity** (a finished shard's blob+index is renamed into place and recorded; a killed run restarts at the next shard) — the current resume is per-image via 137k `.pt` files | 6 h run on a killable box | S | none |
| W7d | **Drop the full-corpus size-scan** (`pack_batches` opens every image to read its dimensions — a whole pass over 138k files) by taking the patch count from the images repo's own metadata or caching it | measurable I/O in the 6 h | S | none |
| W7e | Benchmark the new precompute end-to-end on the target GPU box and **record the number**. Harness: `scripts/bench_precompute.py` (per-stage timing: read / decode / preprocess / h2d / vit / write_pt / write_blob; `--synthetic-old` for the paired old-vs-cap measurement). Target: **≤ 30 min for the full 138,987-image corpus** at the 383 cap | the 6 h run was Colab + Google-Drive + 137k per-image `.pt` writes, not compute: `process_image` alone was 461 ms/img at the old geometry vs 11 ms at the cap (M21), and the CPU side is 3.6–48× the GPU side (M22) | S | none |
| W7f | **Stream shards to HF as they are produced** (one shard blob + index at a time, uploaded immediately): the container's ephemeral disk holds one shard (~2–3 GiB), never the whole corpus. Required because there is **no volume** — the 310 GiB corpus is never materialised on any host | D3 (no Modal volume) | S | none |
| W8 | ~~Re-pack~~ — **merged into W7b**: the precompute emits the final corpus format. Remaining: the global `key_map` (key → shard,row) and the key-index **v4** shape | M7, M15, M16; v4 needed because row→RG is no longer valid | S | **yes** (new plan version) |
| W9 | Publish the new corpus to a **new repo** `keypa/vision-adapter-embeddings-v2` (decided) — a different file format (`.bin`+`.idx`, not parquet) and an unambiguous legacy boundary. Keep the old repo published, tagged **`deprecated`** in its card, so old checkpoints stay resumable | D5, §7 | S | keeps old resumable |
| W9b | **Regenerate the `grid_thw` sidecar** after W6: `train_manifest_grids.jsonl` carries *measured* geometry per row, which becomes stale when the doc slice is re-embedded. `scripts/backfill_grid_thw.py` recomputes it from the image dims using the same `grid_from_dims` → the contract follows the new constant automatically, but the stored file must be rewritten (and `manifest_sha256` changes → new plan version) | M17, `core.py:110-123`, `scripts/backfill_grid_thw.py` | S | **yes** (manifest sha) |

### P2 — The loader (one interface, two backends)

| id | item | why | cost | resume risk |
|---|---|---|---|---|
| W10 | `LocalDiskSource`: whole-object download (hf_transfer) → memmap + index; one shard ahead prefetch | D3; makes the data path ~free where disk allows | M | none (same rows/order) |
| W11 | `RemoteRangeSource`: per-row byte ranges from the same index (kills the 3.19× overfetch) | M7, M9 | M | none (same rows/order) |
| W11b | Make the **plan row-granular** (`build_epoch_plan` today takes whole shards) → fetch time ∝ plan size, and a 300-row plan stops pulling whole shards. This is what makes "1% of the corpus costs 1% of the time" true | M7, D2 invariant | S | **yes** (served stream changes) |
| W12 | Fix the prefetch off-by-one: adopt the *current* row group's bytes by identity (`lo, hi`), not by length; keep the future for the next group so the fetch overlaps training | M8; `stream.py:836-847` compares the next group's length against the current group's size → adopt path is dead | S | none |
| W13 | Make resume skip cheap: don't fetch row groups that contain only rows `< start_pos` | M11 (22.7 min) | S | none (same served stream) |
| W14 | Repair the shard LRU: drop the `max_shards=4` pin; fix the eviction glob (`cache_dir/**/emb_*.parquet` — `hf_hub_download` writes under `data/`) | unbounded disk growth today; disk-fill hazard on any host | S | none |
| W15 | Disk-budget policy: size the cache from `shutil.disk_usage`, fail fast if the next shard does not fit, and *say so* instead of filling the disk. One policy for every host; the *limit* differs (persistent disk vs Modal's ephemeral container disk) | D3 | S | none |
| W16 | **Delete the parquet/Range reader from the hot loop** and replace it with the two `RowSource` impls. The remote path (per-row ranges) is the *Modal* path — first class, not a fallback; the local path is for boxes with a persistent disk | §2.2, D3 | S | none |

### P3 — Re-run the probe (the actual milestone)

| id | item | gate |
|---|---|---|
| W17 | Re-run the probe on the new corpus with the W4 recipe; log `wait_ms` vs `step_ms` | `Σ wait_ms` < 5% of `Σ step_ms`; and **a grok collapse** (loss cliff) or an explicit statement that none occurred |
| W18 | Held-out-60 eval on the final checkpoint (was queued in NEXT_STEPS §1) | numbers recorded in `runs.jsonl` |
| W19 | Hero config (DeepSeek-V4-Flash, B300) — the goal run | the data path is no longer the blocker: on a rental box the corpus is local, on Modal it is streamed (§D3) — *not* "local on both" |

### P4 — Hygiene

| id | item | why |
|---|---|---|
| W20 | Update `docs/DATA.md`: new shard table, n_vis histogram, byte budget, cap rationale | M17–M20 |
| W21 | Record the measurement series in the experiment registry (`runs.jsonl`) with the artifact revision (`probe-checkpoints` repo + `probe_log.jsonl` sha) | steward rule 6 (evidence over vibes) |
| W22 | Fix the attribution in the docs: the method is **Harry Partridge** (a Baseten engineer) — "Baseten's recipe" is right, but the original post and the grok conditions should be cited | §3 D4 |
| W23 | Note in `docs/DATA.md` that the *doc* slice was never pixel-capped and that the `agentic` slice was (the one place the two pipelines differ) | §2 |

---

## 5. Explicitly not doing

- **Not** the corpus rebuild *yet*: it is conditional on W17a's outcome (§2.3).
  The cap is a hypothesis; the collate bug and the unserved 39% are facts.
- **Not** a corpus-wide byte-capped-parquet rewrite (the earlier idea): it needs
  key-index v4, breaks the `row → row_group` invariant, and does *not* reduce
  relative overfetch (density is row-group-size independent). The payload fix
  (W6) removes the need for it.
- **Not** flex/flash-attention work for monster batches: with a per-slice budget
  (D6) there are no monster batches.
- **Not** any 15 GB / 12 GiB-cache memory engineering: those constraints came
  from Colab, which is out of scope (§2.2).
- **Not** copying any corpus or checkpoint locally (steward rule 5).

---

## 6. Reproducing the measurements

Artifacts live in `keypa/vision-adapter-probe-checkpoints`
(`probe_log.jsonl`, `train_4000.log`, `train_manifest.jsonl`).
The replay that produced §1 lives in the session scratchpad
(`/tmp/opencode/*.py`, ephemeral) and is re-derivable from:
the run's `train_manifest.jsonl` + `logs/key_index_cache.json` + the HF shard
listing, with `random.Random(0).shuffle(sorted(shards) − emb_0000/0001)` for
the served order (`train.py:1208` at the run's revision `a020e486`).

---

## 7. Open questions for the human

**Resolved 2026-10-19**

1. **Cap → 383** (W1). Validated reference point; matches the agentic slice;
   uniform batches; corpus ≈ 312 GiB.
2. **Long answers → drop the 645 rows** (W1b): 445 HTML-generation (`websight`)
   + 200 rationale-style = 0.55% of the manifest, the exact category the source
   says "prevents the grokking event" (18.4 GiB of embeddings).
   *How easy is removal across the 3 repos?* **It is a manifest operation.**
   The manifest repo is the authority on what is trained on; the images and
   embeddings repos are content-addressed by `key`, so unreferenced rows simply
   stop being planned, and the next re-pack does not emit them. No edit to the
   images repo is needed. It *does* change `manifest_sha256` → new plan version
   → old checkpoints are not resumable against the new manifest.
3. **Modal volume → none** (D3). Streaming stays first class; disk is an
   opportunistic cache with an explicit limit on every host; nothing may depend
   on it persisting. Backend independence is a hard requirement: same corpus,
   index, plan, row order and results on Modal / Colab / Molab / RunPod.
4. **Old corpus → keep it published, tag it `deprecated` in its card** (W9).
   "Retiring" would mean deleting the current 883.8 GiB
   `keypa/vision-adapter-embeddings` and replacing it in place — which breaks
   `--resume hf` for every checkpoint we hold (step-4000 included), because
   their `plan` metadata pins `manifest_sha256` + `stream_order_hash`, and a
   repack changes the key→row mapping. Deletion is irreversible and buys
   nothing; the cost of keeping is HF storage plus a "legacy" label in the card.
5. ~~Does the images repo store the doc slice at full resolution?~~ **Yes — the
   cauldron rows are NOT capped** (33.80 GiB total; `websight`/`textvqa`/
   `docvqa` mean 0.9–1.1 MB per image, max 25 MB). The agentic rows are capped.
   The images are already in the repo, so **no upstream re-fetch** — but the
   cap must be applied at precompute (W6, one constant). See §2.1, M23.
6. ~~Legacy corpus: keep forever or deprecate?~~ **Keep, and tag `deprecated`.**
7. ~~Precompute scope: all images or only the manifest's rows?~~ **All 138,987
   images** — the corpus is a superset of any manifest; the manifest is the
   cheap, versioned artifact.
8. ~~New repo or new prefix?~~ **New repo** `keypa/vision-adapter-embeddings-v2`.
9. **Precompute host:** benchmark first on the rental box (W7e), then the full
   run on Modal (no volume → W7f streams shards to HF as they are produced).

**Still open**

- **Upload bandwidth for the new corpus**: 310 GiB must go *up* to HF. At
  ~100 MiB/s that is ~53 min of upload — a new cost that the download-side
  benchmarks never measured. `scripts/bench_transfer.py` measures download only;
  an upload bench (or a measured `hf_hub_upload` of one shard) is needed before
  committing to the Modal run.
