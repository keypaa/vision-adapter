# DATA — what the model eats, and why

## What "ETL" means

**Extract, Transform, Load.** In this project:
- **Extract** — pull image bytes out of the upstream datasets
  (`agentsea/wave-ui-25k`, `showlab/ShowUI-desktop`, `xlangai/aguvis-stage2`,
  `HuggingFaceM4/the_cauldron`).
- **Transform** — resize to ≤300k pixels, floor dimensions to 28-px multiples
  (LANCZOS), re-encode PNG/JPEG.
- **Load** — write via `vision_adapter/backends/{local,modal}` into
  `./data/images/...` (or the Modal Volume `/data/images/...`) so every later
  stage reads from one durable, idempotent location.

`vision_adapter/data/agentic.py` + `vision_adapter/data/dataset.py` (with
`vision_adapter/data/cauldron.py` for Cauldron) are the ETL. The dataset
orchestrator also writes the header-first manifest. Everything is resumable:
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
python -m vision_adapter dataset --dry-run --out ./data
```

Under the hood this calls `vision_adapter/data/agentic.py:build_agentic_dataset(backend, ...)`
with `ORDER BY image` determinism.

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
inline in the parquet rows. Pulled via `vision_adapter/data/cauldron.py`
(N_DL=6, N_SAVE=12, `cauldron_manifest.jsonl` contract).

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
./data/images/agentic/<name>.png|jpg        # dataset stage output, 79,659 files
./data/images/cauldron/<subset>-<idx>-<j>.png
./data/metadata/cauldron_manifest.jsonl     # raw cauldron rows
./data/train_manifest.jsonl                 # header-first 45/45/10 SFT mix (trainer input)
./data/train_manifest_val.jsonl             # held-out ~2% for eval hooks
./data/embeddings/<sha1>.pt                 # precomputed MoonViT features (→ packed to shards)
./data/shards/emb_XXXX.parquet              # packed shards: key, n_vis, vis_bytes (SHARD_ROWS=1360, compression=None, per-shard sha256)
./data/logs/train_log.jsonl                 # per-step telemetry (line 0 = config_header, see OPERATIONS.md)
./data/dryrun_report.txt                    # memory-gate verdict
./data/runs.jsonl                           # experiment registry (vision_adapter/registry.py)
```

## Measured `n_vis` distribution (vision tokens per image)

Sourced from `keypa/vision-adapter-embeddings` (`103 shards ×1360 = 136267 rows`
via `vision_adapter/data/stream.py:RemoteShard` `n_vis`-only Range, no
`vis_bytes` on disk; 101 real shards after smoke `emb_0000/0001` excluded;
`vision_adapter/models/preprocess.py:navit_resize` contract
`≤65536 patches, ≤7168 px/side, pad-to-28, 14 px patch, 2×2 merge`).

*Full 101-shard histogram (`136267` rows, `8-way` parallel, `2026-08-30`):*

| `n_vis` bucket | count | share | meaning for GPU |
|---|---|---|---|
| `0-100` | `11094` | `8.1%` | tiny icons `~32-100` → `0.07 MiB` transient, but batched with `4900` → `38 MiB` transient `~500×` swing within one `make_collate` batch — eager `L` tail inflates to `39k` tokens at `bs8` |
| `101-500` | `90961` | `66.8%` | **dominant** — median screenshot `avg 836` → `~400` tokens after `2×2 sd2_tpool`; median batch is medium |
| `501-1000` | `12787` | `9.4%` |  |
| `1001-2000` | `6538` | `4.8%` |  |
| `2001-4900` | `10065` | `7.4%` | large screenshot `~19600 raw → 38 MB` before `2×2 merge` |
| `4901+` | `4822` | `3.5%` | `MAX_PATCHES=4900` overflow — split `RG` `306 MiB` not `450 MiB` (`rg21 54 MiB` observed), `<1%` hit per doc |

`global min 16 max 16653`, `per-shard 20–35 min` — matches precompute
`35→4900` spread flagged as cheapest `10–20%` fragmentation win (see
`docs/research/best-practices.md` and `docs/TRAINING_PLAN.md:31`).

**Why bucketing matters now:**

`vision_adapter/data/stream.py:build_epoch_plan` currently seed-shuffles whole
shards (`Random(0).shuffle`) and takes whole shards greedily until `sample_size`
(`cluster plan: 1959 rows from 101 shards → ~19 rows/shard → ~1 RG/shard` for
`2k`). So one `bs8` batch can be `8×~500` (`4k` tokens, `2 GiB` eager) and the
next `8×4900` (`39k` tokens, `46 GiB` eager before `train.py:_make_chunked_eager`
`2**26` budget), with `GELU 4096→8192` activations swinging `~10×` — `gnorm`
`215→129`, `26s/step` spikes, `peak=12.32 GiB` variance at L4 `bs16` already
measured. The `101-500` majority (`66.8%`) should not pay `4900`'s cost on every
batch. Same fragmentation at precompute: `greedy pack` in input order leaves
slack because small `140-patch` and huge `19600-patch` land in one `patch_cap`.

**Next optimization — landed on `feat/bucketed-hf-streaming@f13ad17` (Phases 0–1 of
`gossamer-launching-minnow.md`):**

* `vision_adapter/data/stream.py` key-index sidecar bumped to `v3` (`shard,row,n_vis`
  per key; `v2` still loads with backward compat). `_one_shard` fetches `key+n_vis`
  together via `rg_span(key)` span. `build_epoch_plan(bucket_by_n_vis=True)` buckets
  shards by median `n_vis` into `0-100/101-500/501-1000/1001-2000/2001-4900/4901+`
  and sorts rows within each shard by `n_vis` so `bs=8` batches are size-homogeneous
  (`66.8%` `101-500` no longer pays `4900`'s `46 GiB` eager).
* `vision_adapter/data/pack.py:_bucket_id` + `bucketed_embedding_order` helpers for
  future bucketed repack (sorted `cache_items` by `n_vis` before `pack_rows`;
  `10–20%` fragmentation gone when repack runs).
* Remaining Phases 2–3 (`hf_transfer` whole-shard on Modal, micro-Range on Colab) and
  Phase 4 (`modal volume delete`) are next — see `gossamer-launching-minnow.md`.

Header-first manifest format (see `vision_adapter/manifest.py`):
line 0 is `{"type":"manifest_header","manifest_version":1,"git_sha":...,"seeds":{...},"upstream":{...},"shard_set_hash":...,"row_count":N}`.
The `ORDER BY image` fix makes the agentic 54k selection deterministic across
rebuilds even if parquet write order drifts.

Embedding filename convention: `sha1(relative_image_path)[:20].pt`, where the
relative path is relative to the `images/` root (`agentic/foo.png`). Identical
on Modal and Colab by construction.
