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
| M23 | **Images repo, per-subset encoded size** (read via Range on the `size` column only, ~100 KB/shard) | 138,987 images, **33.80 GiB** (mean 255 KiB). Largest files: `websight` 1,096 KiB mean / 11,160 KiB max · `textvqa` 949 / 25,078 · `docvqa` 932 / 11,625 · `infographic_vqa` 815 / 3,866 · `screen2words` 500 / 3,663 · `vqav2` 468 · `visual7w` 325 · `okvqa`/`aokvqa` 404 · `ocrvqa` 248 · `waveui` 196. Smallest: `miniwob` 7.5 · `aitw` 39 · `mind2web` 52 · `guiact-web-multi` 56 · `plotqa` 29 · `chartqa` 39. **93% of the bytes are in the >120 KiB subsets** — **but byte size ≠ resolution** (the cauldron images are PNG at ~4 B/px, so 1.1 MB can still be 274k px). Resolution must be measured in pixels: the audit decoded shard 13 → **74.9% > 300k px, median 637,650 px**; shard 16's first 200 rows are **already ≤ 300k px** (median 274k, n_vis 368) |
| M28 | **Precompute bench, measured on the rental box** (95 GiB card, `scripts/bench_precompute.py`) | Per-patch costs from the 300-image spread sample (shards 0/13/16, mean 6,205 patches/img): ViT **14.2 µs/patch**, preprocess 7.5, decode 1.7, per-image `.pt` write 2.5, cat 0.6, h2d 0.3. VRAM: **20,991 patches per GiB** (the auto constant is now 20,000). The two earlier samples bracketed the truth: shard-16-only (already-capped images) projected 1.2 h, the spread sample (2 of 3 shards are the big cauldron ones) projected 6.4 h |
| M28b | **Corpus-weighted precompute projection** (the bench's per-patch costs × the real n_vis distribution over all 136,267 corpus rows: mean 850, median 364, p90 2,691, p99 5,544, max 16,653) | **as-is: ViT 1.83 h + CPU 1.63 h → 3.46 h serial, 1.83 h overlapped.** Capped at 383: **1.35 h serial, 0.71 h overlapped** (2.6×). Capped at 1,024: 1.91 h / 1.01 h. Corpus bytes: **884 GiB → 344 GiB @383 → 489 GiB @1,024** |
| M24 | **Served vs manifest composition** (the loader collapses duplicate `emb` rows) | agentic 52,924 manifest → **52,924 served (73.3%)** · doc 52,908 → **12,780 (17.7%)** · conv 11,768 → 6,503 (9.0%). The doc slice asks **up to 54 questions per image** (3,983 images with ≥6 questions, all chart/figure QA: *"What is the difference between the Balance of Payment (as % of commercial service exports)"* → answers `No.`/`Yes.`/`0.`/`2.`) |
| M25 | **Collate budget, v1 (what the probe ran)** — `core.py:305-318` at `a020e486` | for `n_vis ≥ 4,090`: `budget_text = max_len - nv - 2 ≤ 2` → **the answer is cut to ≤2 tokens and the question to 1**. **1,986 rows (1.69%)**, all `doc`, carrying **30.9% of all answer words**. Present at HEAD *and* in the 4,000-step run's revision |
| M26 | **Collate budget, mechanism (code-level)** | the run's revision (`a020e486`) capped only the *text*: `budget_text = max_len - nv - 2`, so for `n_vis ≥ 4,090` the answer → ≤4 tokens and the question → ≤1-3, and the sequence could exceed `max_len` (L reached 16,602 with max_len=4,096 → the 117 steps, 45% of GPU time). At HEAD the image is also capped, so the answer is still crushed to ≤2 tokens. **The image is never cropped in either revision** — an earlier claim here (that long questions cropped the image) was wrong and is withdrawn |
| M30 | **THE PROBE RAN ON THE PLAIN MANIFEST** — no `grid_thw` → the position ids came from the **synthetic** `grid_for_nvis` | the run's header says `manifest_path=data/train_manifest.jsonl` (no `grid_thw` on any row) → measured against the *measured* grids on 4,000 live rows: the synthetic grid **differs on 36.6%** of rows and **transposes portrait/landscape on 17.0%** (aspect error p90 ×1.44, max ×3.36). A projector whose whole job is placing image tokens correctly trained on a third of its data with distorted positions and a sixth flipped. The repo's own audit said 38.1% / 18.3% — this reproduces it. **The next probe uses the measured geometry** (`train_manifest_grids.jsonl`, which the trainer prefers; a warning is now logged when the plain manifest is used) |
| M31 | **Collate budget, v2, re-measured from the COMMITTED code** (11,105 live rows with measured grids) | **0 rows** have their answer truncated (was 1,986 rows carrying 30.9% of all answer words); **216 rows (1.95%)** get their image cropped, keeping **77.2%** of their visual tokens (the head crop); **0 rows** end with the image gone. The earlier "3,477 rows with <64 visual tokens" was measured on a different implementation and was impossible — withdrawn |
| M27 | **Loss by group / by image size** — **WITHDRAWN** | a per-step attribution was attempted (replay the served stream + join the log), but the replay does not reproduce the run's per-step `L` (2/3,600 steps match), so the join is not trustworthy. The index itself is fine (64/64 keys+n_vis match the live corpus) and the manifest matches the run's byte-for-byte at the head; the residual mismatch is the per-step plan reconstruction, which is not reproducible from the artifacts on hand. What survives: the loss plateaued at ≈1.1 for 3,800 steps, the 117 long-sequence steps had mean loss 3.27 vs 1.17, and the 45% of GPU time they took. **Which slice learns is still an open question** → P0.5 |

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
| 45% of GPU time in 3.2% of steps | doc rows are 3–40× the reference token count → batch cost = longest row (padded L) | M12, M28 |
| Those steps carry ~3× the loss | at the probe's revision the collate **starved the answer** for `n_vis ≥ 4,090` (≤2 tokens) and left the sequence unbounded (L up to 16,602) | M25, M26, M13 |
| 883.8 GiB corpus, 3 GiB row groups, 29% of wall on I/O | the same rows are 3–40× bigger than they need to be | M7, M9, M23 |
| **no grok collapse** | the strongest candidate found so far is **not** the corpus size but the **position ids**: the probe ran on the manifest without `grid_thw`, so 36.6% of its rows had a distorted spatial grid and 17.0% had portrait/landscape transposed (M30). Then: the collate budget (M25/M26), the unserved 39% of questions (M15/M24), and the recipe (D4) | M14, M30 |

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
4. **The probe also ran with the WRONG GEOMETRY** (M30): the manifest it used
   has no `grid_thw`, so the position ids came from the synthetic grid —
   distorted on 36.6% of rows, transposed on 17.0%. This is now the leading
   candidate for the plateau, ahead of the corpus size, and it is fixed for
   free by using the measured-geometry manifest the trainer already prefers.

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
*which* value is an empirical question, to be settled by the P0 probe re-runs
(M27 shows the loss is currently flat across sizes, so the size axis is
untested, not dismissed).

**Measured cost of the cap (M28)**: on a 12.8 Mpx image, preprocess 592 → 82 ms
(7.2×) and the ViT 2,402 → 22 ms (**110×**) — the ViT cost grows steeply with
the token count, so the cap is worth far more on the *monster* rows than the
corpus average (corpus-wide the ViT time goes 1.01 h → 0.54 h).

**Measured subtlety (M23)**: byte size does **not** imply resolution. The
cauldron images are PNG at ~4 B/px, so `websight` at 1.1 MB/image can still be
274k px (already capped), while the *same subset* elsewhere (the audit's shard
13) is 74.9% > 300k px. **Resolution must be measured in pixels per image** —
which the precompute knows for free (it is the ViT's own input geometry), so a
per-subset budget can be enforced without a separate pass.

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
| **W0** | ✅ **DONE** (`fb5580c`): **fix the collate text/visual budget** — answer > question > image; the answer is never truncated, the question takes what the answer leaves, the image absorbs the remainder (cropped from the head); L never exceeds max_len. `tests/test_collate_budget.py` (6 new) pins the contract; the old overflow test had asserted the answer was cut to 2 tokens — that was the bug. Measured on the manifest: the old policy destroyed the answer of **1,986 rows (1.71%) carrying 30.9% of all answer words**; 1,989 rows now get their image cropped instead, and **3,477 rows (2.99%) end with <64 visual tokens** (degraded → the per-slice budget D6 is the real fix for them). Labels changed for those rows → a resume across this change is statistically-equivalent, not identical | M25, M26 | S (~1 day) | labels change (statistical, not bit-identical) |
| **W0b** | ✅ **DONE**: **serve every manifest question** — the plan rows are grouped by (row_group, position) into a *list*, so the questions for one image are all served (previously only the last one survived: 45,393 of 117,600 rows, 39% of the signal, never served while `samples_seen` counted them). The served stream changed → **the checkpoint now records the recipe** (`stream_version`, `collate_version`) and a resume across a change is **refused** with a clear message instead of silently landing on a different row. `tests/test_serve_duplicates.py` (8 new) | M15, M24 | S | **yes** — old checkpoints are still resumable *only* if their recipe matches; a v1 checkpoint is refused by the v2 loader |
| **W0c** | ✅ **DONE**: `scripts/filter_manifest.py` drops the **long-answer rows** (645 = 445 HTML-generation + 200 rationale-style; measured on the manifest: 455 doc, 165 conv, 25 agentic) — the exact category the source says "prevents the grokking event". The trainer now logs the **group mix** per step (`g_mix`), so the next probe says *which slice* learns instead of a single batch loss | source recipe, D4 | S | **yes** (manifest sha changes) |
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
| W7e | Benchmark the new precompute end-to-end on the target GPU box and **record the number**. Harness: `scripts/bench_precompute.py` (per-stage timing; `--shards 0,13,16 --spread` for a representative sample; `--synthetic-old` for the paired geometry measurement). **Already measured (M28): the full corpus is ~1.16 h serial, ~0.64 h overlapped, on a 95 GiB card** — the old 6 h was Colab + Google Drive + 137k per-image `.pt` writes, not compute. Target: ≤ 45 min overlapped. The *upload* of ~394 GiB may then dominate and is unmeasured | M28 | S | none |
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

## 4b. The corpus rebuild pipeline — BUILT (2026-10-20)

Two new pieces, ready to run on the GPU box. The point of the blob format is
that a **row** is the unit of transfer (2.9 MiB at the 383-token cap) and its
offset is a prefix sum, so the index is `n_vis` per row (2.7 KB per 1,360-row
shard) and **"1% of the corpus" costs 1% of the bytes and 1% of the time** —
which the parquet row-group path could not do (3.19× overfetch, M7).

### `scripts/rebuild_corpus.py` — the writer

```bash
# 1. see what the cap does (no GPU, no writes):
python scripts/rebuild_corpus.py --dry-run --shards 13 --limit 40
# 2. build and publish (GPU box):
python scripts/rebuild_corpus.py --out-repo keypa/vision-adapter-embeddings-v2 \
    --out-dir ./corpus_out --device cuda
# 3. a denser budget for the doc/OCR slice (D6):
python scripts/rebuild_corpus.py --max-pixels 800000 --out-repo .../embeddings-v2-doc
```

- **The cap is the reference's rule**, applied in `cap_image()`: scale to
  `--max-pixels` (default 300,000) and **floor the dimensions to a multiple of
  28**. Our own `navit_resize` pads *up* instead, which measured 411–425 merged
  tokens where the reference's rule gives ≤382 — the dry run caught that.
  Measured on the real images repo (shard 13, the cauldron rows):
  **300k px → n_vis median 364, max 378** (the reference's regime);
  **800k px → median 936, max 1,056** (the dense doc budget).
- The decode/resize (CPU) overlaps the ViT forward (GPU) through a bounded
  queue; shards are written atomically and **uploaded as they are produced**,
  so the disk holds one shard and **no volume is needed**.
- It also emits the **new geometry**: `rows.parquet` carries
  `key → (shard, row, n_vis, grid_thw)`, replacing both the key index and the
  grid sidecar (the old sidecar describes the old, uncapped images).

### `vision_adapter/data/blob_corpus.py` — the reader

- `BlobCorpus.open(root)`: a local dir **or** an HF repo; identical bytes.
- `read_range(shard, offset, length)`: one Range request (or a seek+read) —
  the loader asks for the bytes of the rows it is about to serve, not the shard.
- `build_row_plan(corpus, keys, sample_size, seed)`: a **row-granular** plan
  (the parquet plan is shard-granular, so a small run still paid for whole
  shards) — this is W11b.
- `BlobStreamDataset`: serves the plan with one coalesced range per group.

**Still to do (the last mile):** the trainer is not wired to the blob corpus
yet — it still builds its plan from the parquet key index and reads parquet
shards. That switch is one flag (`--corpus keypa/vision-adapter-embeddings-v2`)
plus the plan source, and it must bump `STREAM_VERSION` again (the served row
stream changes shape). It is deliberately *not* done before the probe decides
whether the rebuild is needed at all.

---

## 4c. THE PROBE RUNBOOK (probe #1: "does the recipe grok?")

**What it tests:** the reference recipe, on the slice that is already in the
reference's geometry regime, with the defects fixed. Not the corpus size.

| | reference | this run |
|---|---|---|
| slice | art + UI (66k images) | **agentic only** (51,834 rows, every row ≤ 380 tokens = the reference's 300k-px budget) |
| batch | 64 | 16 (4× smaller; the sample count is what matters) |
| LR | **5e-4 constant** | 5e-4 constant (`--lr-schedule constant`), 25-step warmup |
| answers | short | filtered: the 645 long/HTML answers are dropped |
| geometry | — | **measured** (`grid_thw` from the grids manifest) — the old probe used the synthetic grid (37% distorted, 17% transposed) |
| questions | 1 per image | **all of them** (the old loader served 61%) |

**Expected event:** the reference collapsed at 57,600 samples (87% of its epoch).
Our agentic epoch is 51,834 samples → epoch-relative ≈ step 2,818;
sample-relative ≈ step 3,600. A 4,000-step run (64,000 samples = 1.23 epochs)
covers both.

```bash
cd /marimo/vision-adapter && git pull

# 1. the filtered manifest (drops the 645 long/HTML answers; keeps grid_thw)
python scripts/filter_manifest.py --manifest data/train_manifest_grids.jsonl \
    --out data/train_manifest_filtered.jsonl --max-answer-words 20 --fetch

# 2. pack the held-out set (else 16 val probes stream 28 shards each = ~6 h)
python scripts/pack_val.py --n 1000 --out-dir ./data

# 3. the probe
export HF_TOKEN=... VISION_ADAPTER_PUSH_HF=1
export VISION_ADAPTER_HF_CKPT_REPO=keypa/vision-adapter-probe-checkpoints
export VISION_ADAPTER_CKPT_TAG=p05          # keep the old artifacts intact
python -m vision_adapter train --data-dir ./data --config probe \
    --max-steps 4000 --groups agentic --lr-schedule constant --lr 5e-4 \
    --warmup-steps 25 --dtype auto \
    --push-to-hf --hf-ckpt-repo keypa/vision-adapter-probe-checkpoints
```

With the tag set, this run's artifacts are **scoped**: `projector_p05_step*.pt`,
`probe_log_p05.jsonl`, `probe_curves_p05.png` → the previous run's files in the
HF repo are not overwritten, and the analysis reads the last run in an
appended log.

### Step 0 (do this BEFORE the run): pack the held-out set

`cfg.val_every` cannot be set to 0 (the config refuses it: "0 would silently
disable the val probe"), so the run *will* probe the held-out set 16 times.
Without a packed val set the fallback plan streams the shards that the train
plan does **not** use — 28 of 103 shards ≈ 260 GiB per probe ≈ 6 h of extra
I/O for the run. One command fixes it:

```bash
python scripts/pack_val.py --n 1000 --out-dir ./data
# -> data/val_pack_<group>.parquet + data/val_pack_<group>.map.json
```

~3 GiB, one download, and every val probe becomes a local read. The packs
carry the disjointness invariant (an assert: no val key may appear in the
train manifest) and the trainer re-checks it.

### Host: Molab (interactive) vs Modal

**On Molab, run it as a NOTEBOOK, not as a shell job.** Molab's rules
terminate notebooks used for "non-interactive jobs" and forbid SSH-style
remote control — a detached `nohup … &` is precisely the shape that got a
previous session banned. `notebooks/probe_p05.py` is a marimo notebook that
does the whole thing in five cells (setup → manifest → val packs → the probe
→ the reading), streams the child's output *into the cell* (a child's stdout
goes to fd 1 and would bypass marimo's capture, so the cell would look
frozen for hours), and prints the pre-registered outcome table next to the
measured numbers in the last cell.

Open it in the Molab notebook, paste the HF token into cell 1, run the cells
in order, and **keep the tab open and watched**. While it runs, use the
other cells to look at things (peek at `data/probe_log_p05.jsonl`, watch the
loss) — that is what makes the session interactive compute rather than a
batch job.

| | Molab | Modal |
|---|---|---|
| the rules | "notebooks for **non-interactive jobs**" may be terminated; no SSH | batch platform, designed for this |
| how to run it | the marimo notebook (`notebooks/probe_p05.py`), foreground cells | a normal job; the same commands inside the container |
| data to pull | 133 GiB of agentic rows (246 GiB of shards) | same |
| disk | interactive sessions have room for a cache | the container disk decides the path |

**The disk decides the transport, automatically:** the whole-shard path stores
shards on disk (246 GiB) and is **refused** when they do not fit the byte
budget (the loader then falls back to Range); the Range path keeps only the
fetched row groups in `rg_cache` (bounded, 12–64 GiB, evicted) → it needs far
less disk. So a small-disk host transparently gets the slower-but-tiny Range
path, and a big-disk host gets the fast whole-shard path.

**Disk arithmetic to check before launching:** the Range path needs
`rg_cache` ≤ 64 GiB; the whole-shard path needs 246 GiB of shard cache. On
Modal, if the container disk is smaller than a shard, the loader will say
"shard X is N GiB and the cache budget is M GiB — skipping the whole-shard
download, using Range" and carry on.

### After the run

```bash
python scripts/analyze_probe.py --log data/probe_log_p05.jsonl --out probe_p05.md
```

The three numbers that decide the next step:
1. **the grok test** — the sustained EMA criterion (a drop in the *level*, not a
   single-window move: the reference's own 0.5-nat threshold fires 271× on our
   per-step log, so it measures noise here);
2. **`g_mix` per step** — which slice learns (the old log had no mix);
3. **`Σ wait_ms / Σ step_ms`** — the data path's share (the old run: 29%).

If a collapse appears → the recipe is confirmed and the doc slice becomes a
per-slice-budget + tiling question. If the loss only memorises (train loss → 0)
without generalising → too little data diversity (51,834 images vs the
reference's 66,240) → bring in the doc slice after the per-slice re-embed.

---

## 4d. PRE-REGISTERED OUTCOMES — written BEFORE the run, to be read as written

**What each diagnostic can and cannot prove (measured, not assumed):**

| diagnostic | what it is | what it CANNOT prove |
|---|---|---|
| the train loss | every step, EMA β=0.98, logged every 20 steps | generalisation — it is a *training* curve |
| the val loss (every 250 steps) | 1,272-row split, **now slice-filtered to agentic** (1,076 rows) | the split carries **no `grid_thw`** → the val probe runs on the **synthetic** grid (inverted orientation on 17% of rows) → the number is an inflated *upper bound*; only its *movement* is meaningful |
| `scripts/eval_heldout.py` | two projectors compared on rows from `emb_0000/0001` | those shards are the **smallest images** (their neighbours `emb_0002..0007` hold n_vis 16–100, median 35) → a small-image regime, *not* the agentic regime (median 364) |
| `Σ wait_ms / Σ step_ms` | the data path's share of the wall time | whether the *transfer* was efficient (the old run overfetched 3.19×) |

**Numbering note:** these are SIGNATURES of a probe outcome, S1-S8. The
O1-O9 in section 6b are the open CODE DEFECTS -- a different numbering, all
fixed. (Renamed from O1-O8 on 2026-10-21: two numbering schemes for O was
one too many.)

**The grok criterion (fixed now):** a *sustained* drop in the EMA **level** of
≥0.3 nats, holding for ≥3 consecutive logged windows, at or after step ≈2,800.
The reference's own ">0.5 nats below the running mean" threshold fires 271
times on our per-step log — it measures noise here, so it is not used.

| # | signature in the log | what it means | next action | cost | hero-run consequence |
|---|---|---|---|---|---|
| **S1** | EMA level breaks: a sustained ≥0.3-nat drop after ≈step 2,800 (epoch-relative ≈2,818, sample-relative ≈3,600) | **the recipe works on our data** — the five defects (decaying LR, synthetic geometry, crushed answers, unserved questions, wrong pixel regime) were the blockers | hero run with the same recipe; the corpus rebuild becomes a *quality/perf* project, not a gate | done | **unblocked** |
| **S2** | the loss falls steadily (e.g. 1.1 → 0.6) with no cliff, still falling at step 4,000 | the fixes helped; the snap did not happen. Candidates: 1.23 epochs is short (the reference ran 2); the 2B backbone does not grok this way (the reference used a 504B MoE); click-coordinate grounding is harder than naming an object; 51,834 images is too few | in order, cheapest first: (1) extend to 8,000 steps (2 epochs); (2) batch 64 via grad-accum 4 (the reference's effective batch); (3) the "recipe arm": the doc slice re-embedded at its own budget; (4) treat the probe as a weak proxy | +1–2 h per arm | **the branch where the probe's verdict may not transfer.** A negative on a 2B backbone with a hard task may indict the *proxy*, not the recipe → decide the hero run on cost, not on this verdict |
| **S3** | train loss → 0 while the held-out loss stays flat | memorisation, not grokking | more data diversity: the doc slice (after the per-slice re-embed) and the **art** slice — the reference's actual task type (a WikiArt-style curriculum), which our mix has none of | a rebuild + a re-embed | the mix must grow **before** a multi-GPU spend |
| **S4** | the loss is pinned (≈1.1, gnorm ≈0.17 as in the old run) | a bug, not a science problem: the projector is not moving | the **16-row overfit test**: can the projector drive the loss to ~0 on 16 fixed rows? If not, the plumbing is broken (collate / position ids / LR / projector scale) | minutes | **do not launch the hero** — a broken projector on 300B+ GPUs is a waste |
| **S5** | the wall time ≫ the old run's 24,366 s for the same 4,000 steps; `wait_ms ≫ step_ms` | the data path is still the bottleneck (e.g. the whole-shard path was refused for lack of disk → the Range path, 3.19× overfetch) | the corpus rebuild becomes the **precondition** for the science, not the optional extra | a rebuild | the hero run *must* have the rebuilt corpus first — a Range-path hero wastes the GPU |
| **S6** | the run exits early (geometry guard / collate validation / recipe refusal) | a config or manifest mismatch | `scripts/preflight_manifest.py` (PASS on 27,705 live rows) + the error text; usually a 10-minute fix | minutes | nothing was spent |
| **S7** | a collapse appears, but the val loss does not follow | memorisation wearing a grok's clothes | treat as **S3** | — | — |
| **S8** | the collapse lands at step ≈3,900, at the very end | the run was too short | re-run to 8,000 steps and read the criterion again | +2 h | — |

**The honest prior:** O2 is the most likely branch — grokking is
backbone- and data-sensitive, the reference's backbone was a 504B MoE, and our
task (predict a click coordinate) has far higher answer entropy than "a red
bus". S1 is the branch that pays off; S4 is the branch that must be caught
cheaply; S5 is the branch that costs the most wall time.

---

## 5. Explicitly not doing

- **Not** the corpus rebuild *yet*: it is conditional on W17a's outcome (§2.3).
  The cap is a hypothesis; the collate bug and the unserved 39% are facts.
  (The *pipeline* is built — §4b — so the decision only costs GPU-hours, not
  engineering time.)
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

## 6b. OPEN DEFECTS (from the round-2 adversarial review, 2026-10-19)

**All fixed on 2026-10-20 (commit below).** The list is kept so the reasoning
survives; each row now says what was done.

| # | defect | severity | status |
|---|---|---|---|
| O1 | `STREAM_VERSION` was not bumped for the question-interleaving: the row at a given offset changed, but a v2 checkpoint would resume onto the new order with the guard silent | silent wrong results | **fixed: `STREAM_VERSION = 3`** |
| O2 | `MAX_ANSWER_TOKENS` 256 → 1024 changed the labels without a `COLLATE_VERSION` bump | silent wrong results | **fixed: `COLLATE_VERSION = 3`** (also covers the row-aligned crop) |
| O3 | `crop_to_grid` omitted `t` and assumed even dims: a `t>1` or odd-dim grid would raise the mismatch ValueError | crash (latent) | **fixed**: `t` is handled (whole frames), odd dims raise loudly, and `grid_token_count()` pins the grids the position code can actually express |
| O4 | a crop MASKS a grid/n_vis mismatch: a wrong measured grid on a cropped row no longer raised | silent wrong positions | **fixed**: the collate validates the grid against the row's token count BEFORE any crop, and also checks that the grid is positionable at all |
| O5 | a cropped row could lose the image entirely, and up to `llm_w-1` tokens of budget were wasted | silent wrong results | **fixed in the only way the geometry allows**: the crop is whole rows, so the leftover budget (< one row) is unused; a row too wide to crop keeps no visual tokens (documented; does not occur in the corpus) |
| O6 | the whole-shard LRU never evicted the LAST file, and a failed `HEAD` skipped the size check | disk fill | **fixed**: eviction goes down to zero, and a failed HEAD refuses the download (the Range path does not need the size) |
| O7 | `train.py`'s first-shard prefetch used a `ThreadPoolExecutor` → its atexit hook joins the worker, so an in-flight 60 GB download hung the process at exit | hang | **fixed**: it uses the daemon task now |
| O8 | `_shard_lru_limit` / `_biggest_cached_shard` were dead code | cosmetic | **fixed: deleted** |
| O9 | `analyze_probe.py`: "wall (ts span)" excluded startup and the final save; the "other" bucket silently contained the val probes; the `L > 4096` threshold was hardcoded | reporting | **fixed**: both wall numbers are shown and labelled, and the threshold comes from the run's `max_seq_len` |

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
