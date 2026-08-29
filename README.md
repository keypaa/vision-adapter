# Vision-Adapter — giving DeepSeek-V4-Flash eyes

Train a small (~67M param) **vision projector** that grafts
[**Kimi K3**'s MoonViT-V2 vision tower](https://huggingface.co/keypa/MoonViT-V2-Standalone)
onto the frozen text backbone of
[**DeepSeek-V4-Flash-0731**](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
yielding an agentic / UI-grounding / document-understanding multimodal model.

Recipe follows the Baseten *"GLM 5.2 with vision"* findings (grokking dynamics):
train **only the projector** (both backbones frozen), short on-policy QA, batch 8,
LR 5e-4, AdamW. See `docs/ARCHITECTURE.md` for the full design and verifications.

```text
image ──[preprocess]──> MoonViT-V2 (401M frozen) ──[2×2 merge]──> projector (67.1M, train)
                                                                          │
DeepSeek-V4-Flash (304B frozen, 155 GiB FP8/int8) <── spliced embeddings ──┘
```

Staged CLI: python -m vision_adapter {dataset,precompute,pack,train,probe}  — see docs/PIPELINE.md for the exact rebuild commands.

```bash
git clone https://github.com/keypaa/vision-adapter.git
cd vision-adapter
pip install -e .            # base: torch, pyarrow, pillow
pip install -e ".[train]"   # training deps: transformers, safetensors, accelerate, sentencepiece
# any CUDA GPU works (T4, L4, A100, 4090…); dataset/pack run on CPU, train/precompute require GPU
```

## Layout

| Path | Role |
|---|---|
| `vision_adapter/` | single package — config, core, manifest, registry, cli, backends, data, models |
| `vision_adapter/cli.py` | sole entrypoint `python -m vision_adapter {dataset,precompute,pack,train,probe}` |
| `vision_adapter/config.py` | frozen TrainConfig + config_header provenance |
| `vision_adapter/core.py` | HourglassProjector + collate + inject + monitors |
| `vision_adapter/manifest.py` | header-first manifest I/O + ORDER BY determinism |
| `vision_adapter/registry.py` | runs.jsonl experiment registry |
| `vision_adapter/data/{agentic,cauldron,dataset,pack}.py` | staged data (positional join, ORDER BY image) |
| `vision_adapter/models/{moonvit,preprocess,precompute}.py` | MoonViT forward + navit_resize + shared precompute |
| `vision_adapter/backends/{base,local,modal}.py` | DataBackend local\|modal |
| `tests/` | 9 test files (collate, pack, preprocess, probe, telemetry, backends, cli, dataset, docs) |
| `docs/PIPELINE.md` | rebuild manual — exact commands per stage |
| `docs/` | QUICKSTART, ARCHITECTURE, DATA, TELEMETRY, OPERATIONS, HF_PUBLISH |

## Data on HuggingFace

| Repo | Contents |
|---|---|
| [`keypa/vision-adapter-images`](https://huggingface.co/datasets/keypa/vision-adapter-images) | full image corpus — 17 parquet shards, 138,987 rows |
| [`keypa/vision-adapter-manifests`](https://huggingface.co/datasets/keypa/vision-adapter-manifests) | train/val manifests (45% agentic / 45% doc / 10% conv), `emb`+`user`+`assistant`+`g` rows |
| [`keypa/vision-adapter-embeddings`](https://huggingface.co/datasets/keypa/vision-adapter-embeddings) | precomputed MoonViT-V2 embeddings as parquet shards (`key`/`n_vis`/`vis_bytes`, BF16 raw) |

The trainer reads the precomputed `.pt` cache on the Modal volume directly
(measured ~0–1 % of step time at batch 8); the parquet shards are the portable,
reproducible copy of the same tensors.

## TL;DR run order

```bash
python -m vision_adapter dataset --out ./data --seed 0
python -m vision_adapter precompute --data-dir ./data --revision <sha>
python -m vision_adapter pack --data-dir ./data --hf-only        # or --only 0:2
python -m vision_adapter train --data-dir ./data --config default --dryrun  # dryrun: CPU-safe, no GPU needed
python -m vision_adapter probe --data-dir ./data                  # alias for train --config colab
# Modal: same, --backend modal
# Legacy shims still at root during transition: modal run modal_train.py::train_dryrun
```

## Tests

```bash
python -m pytest -q  # 61 tests: preprocess, pack/resume, collate/inject, telemetry, backends, cli, dataset, docs lint
```

`testpaths = ["tests"]` in `pyproject.toml`; console script `vision-adapter` (`vision_adapter.cli:main`) installed via `pip install -e .`.

Full step-by-step with exact prerequisites: **`docs/QUICKSTART.md`**.
