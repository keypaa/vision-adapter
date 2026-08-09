# QUICKSTART — from zero to a trained projector

This is the end-to-end, copy-pasteable path. Each step lists what it does, how
long it takes, and how to verify it succeeded.

---

## Prereqs (one-time)

```bash
pip install modal huggingface_hub
modal token new            # opens browser; links your Modal account
# HF CLI already logged in? If not:
#   huggingface-cli login
```

Check your HF identity (needed to push the standalone MoonViT repo in step 0):

```bash
python3 -c "from huggingface_hub import whoami; print(whoami())"
```

---

## Step 0 — Projector source: agentic images + MoonViT weights

These are already done and reused; you only re-run if you change something.

* `extract_moonvit_v2.py` — pulled the 401M MoonViT-V2 weights + config out of
  `moonshotai/Kimi-K3` and pushed them to `keypa/MoonViT-V2-Standalone`.
  *Re-run only if you delete the HF repo.*
* `build_agentic_images.py --dry-run` — verifies the six positional joins are
  still intact; safe, touches no pixels.
  *Re-run before any `etl` if the upstream datasets update.*

---

## Step 1 — ETL: build images on Modal  (≈ 30–60 min, one-off)

Downloads the sources (≈ 23 GB), runs the resize/pad pipeline, writes
`/data/images/{agentic,cauldron}/` + `/data/metadata/cauldron_manifest.jsonl`.

```bash
modal run modal_pipeline.py::etl
```

Verify:

```bash
modal volume ls vision-adapter-data images/agentic | head
modal volume ls vision-adapter-data metadata
```

Expect ~30k PNGs in `images/agentic` + dozens of Cauldron configs in `metadata`.

---

## Step 2 — Mix manifest  (< 1 min)

Samples the exact 45 % agentic / 45 % doc / 10 % conversational split into
`/data/train_manifest.jsonl` (the file the trainer actually consumes).

```bash
modal run modal_pipeline.py::build_train_manifest
```

Verify (counter should print `agentic:54000, doc:54000, conv:12000, total 120000`):

```bash
modal volume cat vision-adapter-data train_manifest.jsonl | head -2
```

---

## Step 3 — Precompute MoonViT embeddings (choose ONE)

Embeddings are per-image `[n_merged, 4096]` BF16 tensors, hashed
`sha1(relative_image_path)[:20].pt`. Same hash convention on both backends, so
results are interchangeable.

### 3a · Modal A100 (fast, ~10–20 min)

```bash
modal run modal_pipeline.py::precompute
```

### 3b · Free Colab T4 (≈ 1–2 h, resumable across sessions)

1. Upload `images/` to Drive: `MyDrive/vision_adapter/images/`
2. Open the notebook described in `../COLAB_PRECOMPUTE.md`, run cells 1 → 3.
3. When done, sync the embedding folder back to Modal. The safest way is to `zip`
   it first (79 659 files, ~6 GB) and unzip inside the Modal Volume:

```bash
cd /content/drive/MyDrive/vision_adapter/embeddings
zip -r /content/drive/MyDrive/vision_adapter/embeddings.zip .
```

Then locally:

```bash
# download embeddings.zip from Drive to ~/Downloads/
mkdir -p /tmp/opencode/embeddings_push
cd /tmp/opencode/embeddings_push && unzip ~/Downloads/embeddings.zip
modal volume put vision-adapter-data ./. /embeddings/
```

4. On Modal, one-line sanity check that both precompute backends produced the
   same filename scheme (each embedding is a valid .pt of shape [n,4096]):

```bash
modal volume ls vision-adapter-data embeddings | head
```

Verify either way:

```bash
modal volume ls vision-adapter-data embeddings | wc -l   # should approach 120k
```

---

## Step 4 — Training (A100-80GB)

**Dry-run memory gate first** — it builds the full LLM + projector, runs one
forward/backward on a batch of 8 and asserts peak VRAM < 70 GiB:

```bash
modal run modal_train.py::train_dryrun
```

Expected tail:

```
[dryrun] loss=… n_trainable=67.1M | mem_alloc=…GiB peak=…GiB budget=70GiB -> PASS
```

(The `MEMORY GATE: PASS` text was shorthand; the script prints `-> PASS`/`-> FAIL` followed by an explicit assertion.)

**Then start training.** 2 epochs × (120 000 / 8) ≈ **30 000 steps**. Grokking is
sample-bound, not step-bound: watch `samples_seen` — the Baseten cliff (≈ step 900
at bs 64) maps to **≈ step 7 200–11 000 at our bs 8**. Loss curve is printed every
20 steps and the full JSON telemetry stream lands in `/data/logs/train_log.jsonl`
(see `docs/TELEMETRY.md`):

```bash
modal run modal_train.py::train
```

Checkpoints land in `/data/checkpoints/projector_step*.safetensors` and
`latest.pt` every 20 steps (loss-tracked).

To fetch the trained adapter locally after the run:

```bash
modal volume get vision-adapter-data checkpoints/ ./checkpoints/
```

---

## Step 5 — Evaluate

Quick smoke CPU test — run a single image through the frozen MoonViT and the
trained projector, eyeball the shapes:

```python
import torch, json
from PIL import Image
from safetensors.torch import load_file
from moonvit import load_moonvit_from_safetensors
from preprocess import process_image

cfg = json.load(open('.cache/vision_config.json'))
vit = load_moonvit_from_safetensors('.cache/moonvit_v2.safetensors', cfg,
                                    device='cpu', dtype=torch.float32)
emb = vit(**process_image(Image.new('RGB', (280, 280))))[0]
proj = load_file('checkpoints/projector_final.safetensors')
print('vit merged tokens:', emb.shape)   # expect [n, 4, 1024]
# then run projector + frozen DeepSeek from modal_train.
```

Full MMMU/UI-grounding eval lives outside this repo — this project only trains
the alignment layer.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MEMORY GATE FAIL` on dryrun | batch_size 8 too big for your Modal A100 SKU → lower `BATCH_SIZE` in `modal_train.py` |
| loss stays ~7 after step 12 000 | grokking may not happen this run; try `LR = 7e-4` and rerun; see `docs/TELEMETRY.md` |
| Colab precompute synced but trainer sees no images | `embeddings/` must be at Volume root: `modal volume put vision-adapter-data ./embeddings/. /embeddings/` |
| Out of RAM in ETL | Cauldron pulls whole configs; raise `memory=` on the `etl` fn |
| Colab session died | just re-run Cell 3; `_already_done()` skips everything already cached |
| `KeyError` in aguvis join | upstream re-indexed; re-run `build_agentic_images.py --dry-run` |

If in doubt, re-run the failing step — every stage is idempotent.
