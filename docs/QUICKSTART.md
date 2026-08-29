# QUICKSTART — from zero to a trained projector

> Clone: `git clone https://github.com/keypaa/vision-adapter.git && cd vision-adapter` — note: folder is **lowercase** `vision-adapter`, not `Vision-Adapter`.

This is the end-to-end, copy-pasteable path. Each step lists what it does, how
long it takes, and how to verify it succeeded.

---

## Prereqs (one-time)

```bash
pip install -e .            # base: torch, pyarrow, pillow
pip install -e ".[train]"   # training: transformers, safetensors, accelerate, sentencepiece
# optional for Modal backend:
pip install modal
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

* `vision_adapter/data/extract_moonvit_v2.py` — pulled the 401M MoonViT-V2 weights + config out of
  `moonshotai/Kimi-K3` and pushed them to `keypa/MoonViT-V2-Standalone`.
  *Re-run only if you delete the HF repo.*
* `python -m vision_adapter dataset --dry-run` — verifies the six positional joins are
  still intact; safe, touches no pixels.
  *Re-run before any `dataset` if the upstream datasets update.*

---

## Step 1 — Dataset: build images + header-first manifest  (≈ 30–60 min, one-off)

Downloads the sources (≈ 23 GB), runs the resize/pad pipeline, writes
`./data/images/{agentic,cauldron}/` + `./data/train_manifest.jsonl` (header-first,
`ORDER BY image`, pinned revisions via `vision_adapter/data/dataset.py`).

```bash
python -m vision_adapter dataset --out ./data
# Modal Volume variant:
# python -m vision_adapter dataset --out ./data --backend modal
```

Verify (local):

```bash
ls ./data/images/agentic | head
head -1 ./data/train_manifest.jsonl | python -m json.tool   # {"type":"manifest_header",...}
```

Or on Modal:

```bash
modal volume ls vision-adapter-data images/agentic | head
modal volume ls vision-adapter-data metadata
```

Expect ~30k PNGs in `images/agentic` + dozens of Cauldron configs in `metadata`.

---

## Step 2 — Precompute MoonViT embeddings (choose ONE)

Embeddings are per-image `[n_merged, 4096]` BF16 tensors, hashed
`sha1(relative_image_path)[:20].pt`. Same hash convention on both backends, so
results are interchangeable.

### 2a · Modal A100 (fast, ~10–20 min)

```bash
python -m vision_adapter precompute --data-dir ./data --backend modal
```

### 2b · Free Colab T4 (≈ 1–2 h, resumable across sessions)

1. Upload `images/` to Drive: `MyDrive/vision_adapter/images/`
2. Open `vision_adapter/models/precompute.py` (Colab path uses `--backend local` with a Drive-mounted `--data-dir`; see `vision_adapter/models/precompute_colab.py` — deprecated shim that re-exports the shared precompute), run cells 1 → 3.
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

Verify either way (local or Modal):

```bash
python -m vision_adapter precompute --data-dir ./data --help   # shows --revision pin, --patch-cap
ls ./data/embeddings | wc -l            # local: should approach 120k
modal volume ls vision-adapter-data embeddings | wc -l   # Modal: should approach 120k
```

---

## Step 3 — Pack embeddings into shards  (< 5 min, resumable)

Packs the 120k `.pt` embeddings into ~100 parquet shards (`SHARD_ROWS=1360`,
`compression=None`, per-shard `sha256`) via `vision_adapter/data/pack.py`.

```bash
python -m vision_adapter pack --data-dir ./data
# HF-only push (no local volume copy):
# python -m vision_adapter pack --data-dir ./data --hf-only
# single-shard range:
# python -m vision_adapter pack --data-dir ./data --only 0:2
```

Verify:

```bash
ls ./data/shards | head
python -c "import pyarrow.parquet as pq; print(pq.read_table('./data/shards/emb_0000.parquet').num_rows)"
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
from vision_adapter.models.moonvit import load_moonvit_from_safetensors
from vision_adapter.models.preprocess import process_image

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
| `MEMORY GATE FAIL` on dryrun | batch_size 8 too big for your Modal A100 SKU → lower `batch_size` in `vision_adapter/config.py` (`TrainConfig`) |
| loss stays ~7 after step 12 000 | grokking may not happen this run; try `LR = 7e-4` and rerun; see `docs/TELEMETRY.md` |
| Colab precompute synced but trainer sees no images | `embeddings/` must be at Volume root: `modal volume put vision-adapter-data ./embeddings/. /embeddings/` |
| Out of RAM in ETL | Cauldron pulls whole configs; raise `memory=` on the `dataset` stage (Modal) |
| Colab session died | just re-run the precompute cell; `_already_done()` skips everything already cached |
| `KeyError` in aguvis join | upstream re-indexed; re-run `python -m vision_adapter dataset --dry-run` |

If in doubt, re-run the failing step — every stage is idempotent.
