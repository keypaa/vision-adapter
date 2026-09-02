# PIPELINE — reproducible rebuild manual

One honest place per stage. Each stage is a `vision_adapter` subcommand
(`python -m vision_adapter <stage>`) that shares `vision_adapter/config.py`,
`vision_adapter/manifest.py`, `vision_adapter/registry.py`, `vision_adapter/core.py`
and `vision_adapter/backends/{base,local,modal}.py`.

Historical note: the former 1800-line monolithic pipeline file has been deleted
and split into the staged modules below (see `docs/ARCHITECTURE.md` §3).

---

## 0. Prereqs

```bash
pip install torch pyarrow pillow huggingface_hub
pip install -e .            # exposes `vision-adapter` console script
# Modal backend (optional):
pip install modal && modal token new
huggingface-cli login
python -m vision_adapter --help
```

All stages log `vision_adapter/config.py:config_header` as JSONL line 0
(`run_id`, `git_sha`, `manifest_sha256`, full `TrainConfig`) and append a
`vision_adapter/registry.py:registry_entry` row to `runs.jsonl` best-effort at
`run_end` (correlated by `run_id`).

---

## 1. Dataset — `dataset` (30–60 min, one-off, resumable)

Builds `./data/images/{agentic,cauldron}/` + header-first
`./data/train_manifest.jsonl` (`ORDER BY image`, pinned revisions).

```bash
python -m vision_adapter dataset --out ./data --seed 0 --limit 54000
# pinned upstream revisions (reproducible rebuild):
python -m vision_adapter dataset --out ./data --upstream-pin 0xSero/glm-vision-sft-mix@<sha> --seed 0
# Modal Volume variant:
python -m vision_adapter dataset --out ./data --backend modal --seed 0
# dry-run (positional-join coverage check, no pixels):
python -m vision_adapter dataset --out ./data --dry-run
```

Verify:

```bash
head -1 ./data/train_manifest.jsonl | python -m json.tool  # {"type":"manifest_header","manifest_version":1,...}
ls ./data/images/agentic | wc -l          # ~79,659
ls ./data/images/cauldron | wc -l         # ~59k
# Modal:
modal volume ls vision-adapter-data images/agentic | head
modal volume ls vision-adapter-data metadata
```

Manifest contract: `vision_adapter/manifest.py:write_manifest_with_header`
writes line 0 header (`manifest_version`, `git_sha`, `seeds {python,numpy,torch}`,
`upstream`, `shard_set_hash`, `row_count`, `created_at`). The agentic 54k slice
uses `ORDER BY image LIMIT 54000` — without `ORDER BY`, `random.seed(0)` alone
is nondeterministic. See `docs/DATA.md`.

---

## 2. Precompute — `precompute` (10–20 min A100, ~1–2 h Colab T4, resumable)

Runs `vision_adapter/models/moonvit.py` + `vision_adapter/models/preprocess.py`
(navIT resize, BF16) and writes `embeddings/<sha1>.pt` (`[n_merged, 4096]`).

```bash
python -m vision_adapter precompute --data-dir ./data --revision <commit-sha>
# Modal A100:
python -m vision_adapter precompute --data-dir ./data --backend modal --revision <commit-sha>
# tuning:
python -m vision_adapter precompute --data-dir ./data --patch-cap 262144 --device cuda
```

Pin `keypa/MoonViT-V2-Standalone` with `--revision` (commit hash, not branch) so
a force-push cannot silently change the tower. The embedding key is
`sha1(relative_image_path)[:20].pt` relative to `images/` — identical on local
and Modal (see `vision_adapter/data/pack.py`).

Verify:

```bash
ls ./data/embeddings | wc -l                         # → 120k
modal volume ls vision-adapter-data embeddings | wc -l
python -c "import torch; print(torch.load('./data/embeddings/$(ls ./data/embeddings | head -1)', map_location='cpu').shape)"
```

---

## 3. Pack — `pack` (< 5 min, resumable)

Packs `embeddings/*.pt` into `shards/emb_XXXX.parquet`
(`SHARD_ROWS=1360`, `compression=None`, per-shard `sha256`).

```bash
python -m vision_adapter pack --data-dir ./data --shard-rows 1360
python -m vision_adapter pack --data-dir ./data --only 0:2        # range
python -m vision_adapter pack --data-dir ./data --hf-only         # HF publish
python -m vision_adapter pack --data-dir ./data --backend modal --hf-only
```

Verify:

```bash
ls ./data/shards | head
python -c "import pyarrow.parquet as pq; print(pq.read_table('./data/shards/emb_0000.parquet').num_rows)"
python -c "from vision_adapter.config import file_sha256; print(file_sha256('./data/shards/emb_0000.parquet'))"
# Modal:
modal volume ls vision-adapter-data shards | head
```

Parquet schema: `key` (`embeddings/<sha1>.pt`, byte-identical to manifest `emb`),
`n_vis` (int), `vis_bytes` (raw BF16 `tobytes()`). `file_sha256` parity is
checked post-write (Volume ↔ HF).

---

## 4. Train — `train` / `probe` (A100-80GB or Colab T4, ~30k steps)

```bash
modal run modal_train.py::train_dryrun   # memory gate: peak < 70 GiB (legacy Volume path)
modal run modal_train.py::train
# staged CLI wrappers (config via vision_adapter/config.py:TrainConfig):
python -m vision_adapter train --data-dir ./data --config default --dryrun
python -m vision_adapter probe --data-dir ./data --max-steps 200
# HF streaming (feat/bucketed-hf-streaming, no Volume required):
HF_HUB_ENABLE_HF_TRANSFER=1 python -m vision_adapter train --data-dir ./data --max-steps 200  # Modal: whole-shard 1 GiB/s
python -m vision_adapter train --data-dir ./data --max-steps 200  # Colab T4: Range fallback (32MiB×8, prefetch)
```

Configs: `default_config()` (bs 8), `probe_config()` (bs 16), `colab_probe_config()` (bs 8).
Telemetry: `./data/logs/train_log.jsonl` (line 0 = `config_header`, last = `run_end`),
`./data/runs.jsonl`, `./data/dryrun_report.txt` — see `docs/TELEMETRY.md`.

On `feat/bucketed-hf-streaming` Modal training no longer mounts `vision-adapter-data 930 GiB` for streaming — HF `hf_transfer` whole-shard download pipelined over `33h` (`7ms/file` warm vs `4ms` Volume, `0.7%` vs `0.4%` of step). Colab Range path is bucketed by `n_vis` (`66.8%` `101-500` majority no longer pays `4900`'s cost). See `docs/DATA.md` `n_vis` distribution.

---

## Timing summary

| Stage | Wall time | Bottleneck |
|---|---|---|
| `dataset` (ETL + Cauldron + manifest) | 0.8–1.5 h + 1.5–3 h + 15–90 s | HF download |
| `precompute` (A100 / Colab T4) | 10–20 min / 1–2 h | ViT forward |
| `pack` (120k → 103 shards @1360) | < 5 min local, ~3.5 min/shard HF push | network |
| `train` (120k, bs 8, 2 epochs) | ~30k steps, ~0.1 it/s (CPU offload) | MoE backward |

Every stage is idempotent — re-running skips completed files/shards.
