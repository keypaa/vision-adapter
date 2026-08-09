# DATA — what the model eats, and why

## What "ETL" means

**Extract, Transform, Load.** In this project:
- **Extract** — pull image bytes out of the upstream datasets
  (`agentsea/wave-ui-25k`, `showlab/ShowUI-desktop`, `xlangai/aguvis-stage2`,
  `HuggingFaceM4/the_cauldron`).
- **Transform** — resize to ≤300k pixels, floor dimensions to 28-px multiples
  (LANCZOS), re-encode PNG/JPEG.
- **Load** — write into the Modal Volume (`/data/images/...`) so every later
  stage reads from one durable, idempotent location.

`build_agentic_images.py` + `modal_pipeline.py::etl` are the ETL. The Modal
`etl` function also downloads the Cauldron subsets. Everything is resumable:
re-running skips files that already exist.

## The two corpora

### 1. Agentic / UI (45% of training mix) — `0xSero/glm-vision-sft-mix`

157k rows of short-answer UI grounding data. We use two of its three subsets:

| Sero source | Rows | Unique images | Teaches | Built from |
|---|---|---|---|---|
| `screenshots` | 32.5k | 30,850 | single-step UI grounding, click coords | wave-ui-25k, ShowUI-desktop |
| `multistep` | 50.3k | 48,809 | multi-step browser/OS actions (UI-TARS space) | aguvis-stage2 |
| ~~`art`~~ | 74k | — | (dropped: classification style, off-task) | WikiArt |

**The images are not hosted** — rows carry only filenames (`waveui_013568.png`).
We regenerate them by a **positional join**: the numeric suffix is the 0-based
index into the upstream record list. Verified for all six prefixes:

| Prefix | Upstream | Upstream size | Sero max index | Join |
|---|---|---|---|---|
| `waveui_` | wave-ui-25k (22 parquet shards, embedded bytes) | 24,978 | 24,977 | ✔ |
| `showui_` | ShowUI-desktop (34 parquet shards, embedded bytes) | 7,496 | 7,495 | ✔ |
| `aitw_` | aguvis `aitw-l1.json` + `aitw.zip` | 18,992 | 18,991 | ✔ |
| `miniwob_` | aguvis `miniwob-l1.json` + `miniwob.zip` | 9,826 | 9,825 | ✔ |
| `mind2web_` | aguvis `mind2web-l1.json` + `mind2web.zip` | 7,591 | 7,589 | ✔ |
| `guiact-web-multi_` | aguvis `guiact-web-multi-l1.json` + zip | 16,704 | 16,702 | ✔ |

If any prefix ever fails the coverage check, the builder **refuses to guess**
(fail-closed `IndexError`). Dry-run before building:

```bash
.venv/bin/python3 build_agentic_images.py --dry-run
```

### 2. General / reasoning (45%) + conversational (10%) — `HuggingFaceM4/the_cauldron`

The Cauldron is a 50-subset mixture of classic VQA/document/diagram datasets in
one uniform format (`images`, `texts[{user, assistant, source}]`).

| Slice | Subsets | Mix target |
|---|---|---|
| Reasoning-doc (45%) | chartqa, docvqa, infographic_vqa, screen2words, websight, ocrvqa, textvqa, plotqa, ai2d, scienceqa | 54k |
| Conversational (10%) | vqav2, okvqa, aokvqa, visual7w | 12k |

Chosen because: licenses are (mostly) permissive, answers are *short and
on-policy* (critical for the grokking dynamics — see ARCHITECTURE.md), and the
document/chart/screenshot subsets transfer directly to coding-agent screenshot
reading. `the_cauldron` needs no separate ETL beyond download — images come
inline in the parquet rows.

| Group | Examples | Share |
|---|---|---|
| agentic | 54,000 | 45% |
| doc | 54,000 | 45% |
| conv | 12,000 | 10% |
| **total** | **120,000** | 100% |

## On "66k vs 120k" — is the recipe size mismatched?

No, but the number needs translating. Baseten's ~900-step grok point was:
66k images × batch 64 → 1035 steps ≈ 1 epoch; grok observed *just before
epoch end*, i.e. after the model had seen **≈ 58k samples**.

Our manifest is 120k, and our per-device batch is 8 (A100 memory-bound).
Grokking correlates with *samples seen*, not wall-clock steps. The honest
translation:

| Recipe | batch | dataset | 1 epoch = | grok window |
|---|---|---|---|---|
| Baseten GLM | 64 | 66k | 1035 steps | ~step 900 |
| Ours | 8 | 120k | 15,000 steps | once ~58k–90k samples seen ⇒ ~7,200–11,000 steps |

`modal_train.py` currently runs `EPOCHS × ceil(N/BATCH)` steps as written (it
logs loss every step, so you watch the curve for the collapse). It does **not**
yet implement samples-seen-based early stopping around the 7–11k window —
that's a deliberate enhancement option, flagged in OPERATIONS.md, not something
to assume is already wired in. The exact grok step is empirical — which is
exactly why we log everything (see OPERATIONS.md).

## Files on disk / Volume

```
/data/images/agentic/<name>.png|jpg        # ETL output, 79,659 files
/data/images/cauldron/<subset>-<idx>-<j>.png
/data/metadata/cauldron_manifest.jsonl     # raw cauldron rows
/data/train_manifest.jsonl                 # the 45/45/10 SFT mix (trainer input)
/data/train_manifest_val.jsonl             # held-out ~2% for eval hooks
/data/embeddings/<sha1>.pt                 # precomputed MoonViT features
/data/logs/train_log.jsonl                 # per-step telemetry (see OPERATIONS.md)
/data/dryrun_report.txt                    # memory-gate verdict
```

Embedding filename convention: `sha1(relative_image_path)[:20].pt`, where the
relative path is relative to the `images/` root (`agentic/foo.png`). Identical
on Modal and Colab by construction.
