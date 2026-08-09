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

## Layout

| Path | Role |
|---|---|
| `moonvit.py` | Standalone MoonViT-V2 forward (matches Kimi bit-exact) |
| `preprocess.py` | navit_resize + 0.5/0.5 normalize + patchify |
| `build_agentic_images.py` | reconstructs agentic screenshots/UI frames from upstream HF sources |
| `modal_pipeline.py` | Modal: ETL → train-manifest → (A100) MoonViT precompute |
| `modal_train.py` | Modal: dry-run memory gate + SFT trainer (A100 80GB) |
| `precompute_colab.py` | Same precompute on a free Colab T4 (resumable) |
| `extract_moonvit_v2.py` | One-off: pulled MoonViT-V2 out of Kimi-K3 |
| `docs/` | Architecture, data, operations, quickstart |

## TL;DR run order

```bash
# 1. extract + publish the vision tower (done; weights on keypa/MoonViT-V2-Standalone)
python3 extract_moonvit_v2.py            # PUSH=1 to re-upload

# 2. build the 79k-image agentic corpus + cauldron manifest on Modal
modal run modal_pipeline.py::etl
modal run modal_pipeline.py::build_train_manifest

# 3a. precompute vision embeddings on the A100 (fast), OR
modal run modal_pipeline.py::precompute
# 3b. … precompute free on Colab T4 across 4h sessions (see docs/QUICKSTART.md)

# 4. memory gate, then train
modal run modal_train.py::train_dryrun   # must print "MEMORY GATE PASS"
modal run modal_train.py::train
```

Full step-by-step with exact prerequisites: **`docs/QUICKSTART.md`**.
