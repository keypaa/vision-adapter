# HF_PUBLISH — what the pipeline uploads to Hugging Face, and how to consume it

The Vision-Adapter project publishes **three** Hugging Face repos. One is a
model repo; two are dataset repos. Everything is opt-in per repo via the
`repo_ns` (namespace) and `public` flags in `vision_adapter/data/pack.py`.

## Repos at a glance

| Repo | Type | Visibility | What it holds |
|---|---|---|---|
| `keypa/MoonViT-V2-Standalone` | **model** | public | 401 M-param BF16 MoonViT-V2 weights, `vision_config.json`, Kimi's own `mm_projector`, and the runtime code (`moonvit.py`, `preprocess.py`, etc.) |
| `keypa/vision-adapter-data` | dataset | public by default | `train_manifest.jsonl` (header-first) + `train_manifest_val.jsonl` + `cauldron_manifest.jsonl` + `shards/emb_*.parquet` (the post-ETL mix; images themselves link back to source via the manifest rows; shards carry per-shard `sha256`) |

The processed image bytes (≈ 25 GB) are deliberately **not** rehosted — they reconstruct deterministically from the upstream sources via `python -m vision_adapter dataset --out ./data` (local or `--backend modal`). If you want a dataset with the actual image payloads, publish the images via the source datasets' native flow; the manifests are what you share for training.

### How consumers use it

### 1. The model repo (already public)

```python
from huggingface_hub import hf_hub_download
from vision_adapter.models.moonvit import load_moonvit_from_safetensors
import json

cfg = json.load(open(hf_hub_download("keypa/MoonViT-V2-Standalone", "vision_config.json")))
sd  = hf_hub_download("keypa/MoonViT-V2-Standalone", "moonvit_v2.safetensors")
vit = load_moonvit_from_safetensors(sd, cfg, device="cpu", dtype=torch.float32)
```

Do **not** add large training checkpoints here — the model repo stays lean.
Training artefacts live on the Volume / `./data`, exported by `train` (see
QUICKSTART and `docs/PIPELINE.md`).

### 2. The dataset repos

Any fresh clone / training run can now skip the ETL entirely:

```python
from huggingface_hub import snapshot_download, hf_hub_download
import json, os

# 79k images (25 GB)
snap = snapshot_download(f"{YOU}/vision-adapter-agentic-images")
print("images root:", snap)

# the 45/45/10 recipe actually used (header-first)
m = json.load(open(hf_hub_download(
    f"{YOU}/vision-adapter-mix-manifest", "train_manifest.jsonl")))
```

### 3. The manifest schema

Header-first: line 0 is `{"type":"manifest_header","manifest_version":1,"git_sha":...,"seeds":{...},"upstream":{...},"shard_set_hash":...,"row_count":N}`.
Data rows:

```json
{"emb": "embeddings/<sha1>.pt", "user": "…", "assistant": "…", "g": "agentic|doc|conv"}
```

The `emb` field is the data-relative path of the precomputed MoonViT
embedding. The hash is `sha1(relative_image_path)[:20].pt` where
`relative_image_path` is relative to `images/` (e.g. `agentic/waveui_000123.png`).
That convention makes the manifest *portable* — you can point training at the
same manifest whether your embeddings came from the Modal A100 or a Colab T4.
Parquet shards mirror this via `key` (`embeddings/<sha1>.pt`) + `n_vis` + `vis_bytes`.

## Publishing

```bash
python -m vision_adapter pack --data-dir ./data --hf-only
# or locally without HF (just build shards):
# python -m vision_adapter pack --data-dir ./data
# single-shard range:
# python -m vision_adapter pack --data-dir ./data --only 0:2 --hf-only
```

That's it. Datasets land under **your** account — the function defaults to
`repo_ns="keypa"`, which matches your `keypa/MoonViT-V2-Standalone` repo.
For a direct local push without staging through shards, see
`vision_adapter/data/pack.py:push_local_pack` (historical `local_pack.py` path).

(Optional knobs, only relevant if someone forks this: `--repo_ns` changes the
account, `--public false` makes the repos private.)

Re-running is safe: only missing or changed files upload; nothing remote is
deleted. Each shard is verified by its `sha256` (see `vision_adapter/config.py:file_sha256`).
