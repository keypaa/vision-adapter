# HF_PUBLISH — what the pipeline uploads to Hugging Face, and how to consume it

The Vision-Adapter project publishes **four** Hugging Face repos. One is a
model repo; three are dataset repos. Everything inside `push_datasets_to_hf`
is opt-in per repo via the `repo_ns` (namespace) and `public` flags.

## Repos at a glance

| Repo | Type | Visibility | What it holds |
|---|---|---|---|
| `keypa/MoonViT-V2-Standalone` | **model** | public | 401 M-param BF16 MoonViT-V2 weights, `vision_config.json`, Kimi's own `mm_projector`, and the runtime code (`moonvit.py`, `preprocess.py`, etc.) |
| `{you}/vision-adapter-agentic-images` | dataset | public by default | the full 79 659-file processed-image corpus (agentic + cauldron), id of original source, original Sero/Cauldron ID, image size |
| `{you}/vision-adapter-mix-manifest` | dataset | public by default | `train_manifest.jsonl` (117 600 train rows) + `train_manifest_val.jsonl` (2 400 val rows) — the exact mix the trainer reads |
| `{you}/vision-adapter-cauldron-manifest` | dataset | public by default | raw `cauldron_manifest.jsonl` (≈ 1.9 M rows) before 45/45/10 sampling — useful if you want a different ratio later |

## How consumers use them

### 1. The model repo (already public)

```python
from huggingface_hub import hf_hub_download
from moonvit import load_moonvit_from_safetensors
import json

cfg = json.load(open(hf_hub_download("keypa/MoonViT-V2-Standalone", "vision_config.json")))
sd  = hf_hub_download("keypa/MoonViT-V2-Standalone", "moonvit_v2.safetensors")
vit = load_moonvit_from_safetensors(sd, cfg, device="cpu", dtype=torch.float32)
```

Do **not** add large training checkpoints here — the model repo stays lean.
Training artefacts live on the Modal Volume, exported by `train` (see
QUICKSTART).

### 2. The dataset repos

Any fresh clone / training run can now skip the ETL entirely:

```python
from huggingface_hub import snapshot_download, hf_hub_download
import json, os

# 79k images (25 GB)
snap = snapshot_download(f"{YOU}/vision-adapter-agentic-images")
print("images root:", snap)

# the 45/45/10 recipe actually used
m = json.load(open(hf_hub_download(
    f"{YOU}/vision-adapter-mix-manifest", "train_manifest.jsonl")))
```

### 3. The manifest schema

```json
{"emb": "embeddings/<sha1>.pt", "user": "…", "assistant": "…", "g": "agentic|doc|conv"}
```

The `emb` field is the Modal-Volume-relative path of the precomputed MoonViT
embedding. The hash is `sha1(relative_image_path)[:20].pt` where
`relative_image_path` is relative to `/images/` (e.g. `agentic/waveui_000123.png`).
That convention makes the manifest *portable* — you can point training at the
same manifest whether your embeddings came from the Modal A100 or a Colab T4.

## Publishing

```bash
modal run modal_pipeline.py::push_datasets_to_hf
```

That's it. All three datasets land under **your** account automatically —
the function defaults to `repo_ns="keypa"`, which matches your
`keypa/MoonViT-V2-Standalone` repo. No arguments needed.

(Optional knobs, only relevant if someone forks this: `--repo_ns` changes the
account, `--public false` makes the repos private.)

Re-running is safe: only missing or changed files upload; nothing remote is
deleted.
