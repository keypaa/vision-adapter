# Running MoonViT-V2 Embedding Precompute on Google Colab (free T4)

This computes the frozen vision-tower embeddings for the whole image corpus on a free
Colab **Tesla T4 (15 GB VRAM)**, with automatic resume — if Colab kills your session
after 4 h, you just re-run the same cell and it continues where it left off.

We already pushed everything needed to the HF repo `keypa/MoonViT-V2-Standalone`,
so you do **not** need to copy code files by hand — a single setup cell downloads them.

---

## 1 · Create a Drive layout on your Google Drive

```
MyDrive/vision_adapter/
├── images/               ← put the corpus here (agentic/*.png, cauldron/*.png)
└── embeddings/           ← the Colab run fills this (safe; it's the checkpoint)
```

`images/agentic/` gets the ETL output from the Modal `etl` stage (or the local
`build_agentic_images.py` run). `embeddings/` starts empty; Colab fills it.

> The image *files* are ~24 GB. Uploading is the slow part; run the ETL on the same
> machine that has the images, or export them from the Modal Volume.

---

## 2 · Colab notebook — paste these cells

**Cell 1 — environment**

```python
import os, glob, time
!pip install torch safetensors pillow numpy huggingface_hub accelerate --quiet
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 — fetch code + set paths**

```python
import os, urllib.request, base64, json
from huggingface_hub import hf_hub_download

MOONVIT_REPO = "keypa/MoonViT-V2-Standalone"
os.makedirs("/content/va", exist_ok=True)
for fn in ["moonvit.py", "preprocess.py", "precompute_colab.py"]:
    p = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model", filename=fn)
    dst = f"/content/va/{fn}"
    open(dst, "wb").write(open(p, "rb").read())
    print("fetched", fn, "->", dst)

import sys
sys.path.insert(0, "/content/va")
```

**Cell 3 — configure & run**

```python
import precompute_colab as pc

pc.IMAGES_ROOT = "/content/drive/MyDrive/vision_adapter/images"   # your Drive corpus
pc.OUT_ROOT    = "/content/drive/MyDrive/vision_adapter/embeddings"  # persists across sessions
pc.BF16        = True
pc.BATCH_PATCHES = 240_000   # T4-safe; raise to ~400_000 if you have headroom
pc.FLUSH_EVERY   = 2_000
pc.run()                     # safe to re-run after a Colab disconnect/skips cached
```

---

## 3 · What to expect on a T4

| Metric | T4 (15 GB) | Note |
|---|---|---|
| Model weights | 0.80 GB bf16 | frozen |
| Activation headroom | ~10 GB | sets `BATCH_PATCHES` |
| Throughput (est.) | ~25–60 img/s | depends on mean image size |
| ~120 k corpus ETA | **~0.6 – 1.3 h** | single Colab session |

If Colab times out at 4 h mid-corpus: re-run Cell 3. `_already_done()` skips images whose
4096-dim embedding file already exists and decodes — no duplicate work.

Monitor VRAM inside the run (it prints `gpu=…GB peak=…GB` every 250 images); if `peak`
gets close to 14 GB, lower `PC.BATCH_PATCHES` to 160_000.

---

## 4 · After precompute finishes

`embeddings/` now holds `*.pt` files named
`sha1("<relative/path/inside/images>")[:20].pt` — e.g. the hash of
`agentic/waveui_000123.png`, identical to what `modal_pipeline.py::precompute` would
produce for the same logical image.

Two ways to consume it from the training side:

1. **Upload Drive → Modal Volume** (recommended): from a machine with `modal` CLI, run
   from the folder that contains your local `embeddings/` dir:
   ```bash
   modal volume put vision-adapter-data ./embeddings/. /embeddings/
   ```
   The first arg is the Volume name; `./embeddings/.` copies the **contents** of local
   `embeddings/` into the Volume's `/embeddings/` directory. Then confirm:
   ```bash
   modal run modal_train.py::train_dryrun   # must print "MEMORY GATE: PASS"
   modal run modal_train.py::train         # only after PASS
   ```
2. Keep on Drive and let the *Colab* notebook also do a small CPU-side projector
   sanity pass (not full training — the frozen 155 GiB DeepSeek needs the A100.

---

## 5 · File/check summary

| What | Where | Why |
|---|---|---|
| Weights | HF `keypa/MoonViT-V2-Standalone` | fetched by Cell 2 |
| Config | same repo (`vision_config.json`) | dims + merge params |
| Code | same repo (`moonvit.py`, `preprocess.py`, `precompute_colab.py`) | fetched by Cell 2 |
| Corpus input | Drive `MyDrive/vision_adapter/images/` | you / ETL upload |
| Embeddings out | Drive `MyDrive/vision_adapter/embeddings/` | phase-4 training input |
