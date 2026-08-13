#!/usr/bin/env python3
"""
modal_pipeline.py — Vision-Adapter Phase-3 pipeline for Modal.

One app, persistent Volume. Stages (idempotent + resumable):
  etl                 : build ~79,659 agentic images (positional join into
                        wave-ui / ShowUI / aguvis) + download Cauldron subsets
                        for the 45/45/10 mix -> writes images + metadata to the Volume.
  build_train_manifest: emit /data/train_manifest.jsonl in the exact schema
                        modal_train.EmbSFT reads (45% agentic / 45% doc / 10% conv).
  precompute          : run frozen MoonViT-V2 over every image and cache the
                        4096-dim merged projector-input embeddings (BF16 .pt) to the Volume.
                        (Alternatively run precompute_colab.py on a free Colab T4 and
                        upload the resulting embedding directory back to the Volume;
                        the manifest hash convention is identical so both interoperate.)

Phase-4 training lives in modal_train.py — run:
  modal run modal_train.py::train_dryrun      # hard 70GiB VRAM gate
  modal run modal_train.py::train             # only after dryrun prints MEMORY GATE PASS
"""
from __future__ import annotations
import os
import modal

# ------------------------------------------------------------------ infra ---
APP_NAME = "vision-adapter"
VOL_NAME = "vision-adapter-data"
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
VOLUME_DIR = "/data"
IMG_DIR = f"{VOLUME_DIR}/images"          # agentic + cauldron images (jpg/png)
EMB_DIR = f"{VOLUME_DIR}/embeddings"      # moonvit merged embeddings, .pt cache
META_DIR = f"{VOLUME_DIR}/metadata"       # mix manifests, dataset indices
CKPT_DIR = f"{VOLUME_DIR}/checkpoints"

HF_CACHE = "/hf"                           # model weight cache on a second volume
hf_vol = modal.Volume.from_name("vision-adapter-hf", create_if_missing=True)

MOONVIT_REPO = "keypa/MoonViT-V2-Standalone"
DS_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"

# requested by phase-1 spec
GPU = "A100-80GB"
GPU_MEM_BUDGET_GIB = 70.0                  # max_memory={0:...}
SYS_RAM_BUDGET_GIB = 200.0
MAX_SEQ_LEN = 4096
BATCH_SIZE = 8

class _Const:
    BATCH = 16  # micro-batch of images per packed precompute forward

import os as _os

def _local_hf_token():
    t = (_os.environ.get("HF_TOKEN")
         or _os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not t:
        p = _os.path.expanduser("~/.cache/huggingface/token")
        if _os.path.exists(p):
            t = open(p).read().strip()
    if not t:
        raise RuntimeError(
            "No HF token found. Set HF_TOKEN env var or run `huggingface-cli login` "
            "once on this machine so Modal can forward it into the container.")
    return t


_hf_env = {"HF_HOME": HF_CACHE, "HF_TOKEN": _local_hf_token()}

_etl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("duckdb", "pyarrow", "pillow", "requests", "huggingface_hub", "datasets")
    .env(_hf_env)
)
# FlashAttention wheel must ABI-match the torch build.  Pin torch==2.6.0 and use
# the matching Dao-AILab wheel (cp311 = py3.11) so moonvit.py's varlen path runs.
# Without flash-attn the EncoderLayer falls back to a per-image Python loop over
# F.scaled_dot_product_attention, which underutilizes the A100 badly.
_FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/"
                     "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-"
                     "cp311-cp311-linux_x86_64.whl")
_vit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "safetensors", "pillow", "numpy", "huggingface_hub", "accelerate")
    .pip_install(_FLASH_ATTN_WHEEL)
    .env(_hf_env)
)

app = modal.App(APP_NAME)


# =====================================================================
# STAGE etl
# =====================================================================
@app.function(image=_etl_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=60 * 60 * 6, ephemeral_disk=524288)  # 512 GiB (Modal floor for this SKU)
def etl():
    # Short-circuit: if the volume already holds the final agentic corpus (this is
    # what stays when Modal preempts us mid-run), skip the whole build phase and
    # jump straight into Cauldron — no point re-downloading 24 GB to re-derive
    # the same ~80k files.
    vol_img = os.path.join(VOLUME_DIR, "images", "agentic")
    n_existing = 0
    if os.path.isdir(vol_img):
        n_existing = sum(1 for f in os.listdir(vol_img) if os.path.isfile(os.path.join(vol_img, f)))
    skip_agentic = n_existing >= 75_000
    if skip_agentic:
        print(f"[etl] agentic corpus already on disk ({n_existing} images) — skipping agentic phase")
    import sys, json
    sys.path.insert(0, "/root")  # we'll mount uploaded modules
    from huggingface_hub import hf_hub_download
    os.makedirs(IMG_DIR, exist_ok=True); os.makedirs(META_DIR, exist_ok=True)

    # pull the pipeline modules from the MoonViT repo so this container has them
    for m in ["build_agentic_images.py", "preprocess.py"]:
        p = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model", filename=m)
        with open(p) as f: open(f"/root/{m}", "w").write(f.read())

    # ---- agentic: reuse the verified builder, but pointed at the Volume ----
    import build_agentic_images as bai
    bai.MANIFEST = "/root/sero_manifest.parquet"
    bai.DEFAULT_OUT = f"{IMG_DIR}/agentic"
    os.makedirs(bai.DEFAULT_OUT, exist_ok=True)

    if not skip_agentic:
        # fetch the sero manifest (conversation rows) — a small parquet of unique image names
        mp = hf_hub_download(
            repo_id="0xSero/glm-vision-sft-mix", repo_type="dataset",
            filename="sero_manifest.parquet") if _hf_has("0xSero/glm-vision-sft-mix", "sero_manifest.parquet") else _build_manifest_local()

        print("[etl] building agentic images (direct-from-source positional join) ...")
        for subset in bai.ALL_SUBSETS:
            print(f"[etl] subset={subset}")
            bai_build_one(subset, bai)   # runs bai for that subset into IMG_DIR/agentic
        vol.commit()

    # ---- cauldron: download permissive subsets, split doc/conversational ----
    print("[etl] downloading Cauldron permissive subsets ...")
    cauldron_pull()
    vol.commit()
    print("[etl] DONE.")


def _hf_has(repo, filename):
    from huggingface_hub import file_exists
    try:
        return file_exists(repo_id=repo, repo_type="dataset", filename=filename)
    except Exception:
        return False


def _build_manifest_local():
    import duckdb, os
    os.makedirs("/root", exist_ok=True)
    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("""
      COPY (SELECT DISTINCT image, source FROM read_parquet(
        'https://huggingface.co/datasets/0xSero/glm-vision-sft-mix/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet')
        WHERE source IN ('screenshots','multistep'))
      TO '/root/sero_manifest.parquet' (FORMAT PARQUET)""")
    return "/root/sero_manifest.parquet"


# =====================================================================
# STAGE push_datasets_to_hf
# ============================ dataset publishing ============================
# Goal: once the ETL finishes, push the processed datasets to HF so others can
# reuse them without rerunning the pipeline.
#
# Two publication targets:
#  * push_image_corpus_to_hf — 79k-images encoded as HF Datasets parquet shards
#    (the actual binary corpus, ~25 GB). Packed into ~4 GB shards to match HF's
#    dataset file-size preferences.
#  * push_mix_manifest_to_hf — manifests + cauldron metadata (the 45/45/10 recipe).

_datasets_tmp = "/tmp/agentic_images.parquet"
_datasets_card = "/tmp/vision-adapter-images-README.md"


@app.function(image=_etl_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=60 * 60 * 8, memory="32GB",
              ephemeral_disk=524288)  # 512 GiB, matches etl()
def push_image_corpus_to_hf(repo_ns: str = "keypa", shard_rows: int = 8192):
    """Pack the processed agentic + cauldron images into a single HF dataset.

    The corpus (~25-40 GB of raw PNG/JPEG bytes across ~145k files) is streamed
    row-by-row, so memory stays flat: instead of materialising every image in
    RAM, each parquet shard is written then uploaded immediately.  Shards that
    already exist on the Hub are skipped, so an interrupted push resumes instead
    of restarting.  (The original build loaded the whole corpus into a single
    pandas DataFrame inside a 16 GB container and then uploaded the folder — it
    thrashed/OOM'd and repeatedly blew the old 1h timeout.)

    Each row is {image: bytes, filename: str, source: 'agentic'|'cauldron', size: int}.
    Compatible with `datasets.load_dataset`.
    """
    import os, json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    vol.reload()
    repo_id = f"{repo_ns}/vision-adapter-images"
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    remote = set(api.list_repo_files(repo_id, repo_type="dataset"))

    # enumerate all corpus files once so we know the shard count up front
    files = []
    for grp in ("agentic", "cauldron"):
        grpdir = os.path.join(VOLUME_DIR, "images", grp)
        for fn in sorted(os.listdir(grpdir)):
            files.append((f"{grp}/{fn}", os.path.join(grpdir, fn)))
    total = len(files)
    n_shards = (total + shard_rows - 1) // shard_rows
    print(f"[image-push] corpus rows: {total}  shards: {n_shards}")

    data_rel = "data"
    shard_dir = os.path.join(_datasets_tmp, data_rel)
    os.makedirs(shard_dir, exist_ok=True)

    schema = pa.schema([
        pa.field("image", pa.binary()),
        pa.field("filename", pa.string()),
        pa.field("source", pa.string()),
        pa.field("size", pa.int64()),
    ])

    def shard_name(i):
        return f"train-{i:05d}-of-{n_shards:05d}.parquet"

    import time
    from concurrent.futures import ThreadPoolExecutor

    def read_chunk(chunk):
        """Read one shard's images off the Volume in parallel.

        Modal Volumes are network-backed: per-file open/read has ~50-100 ms
        round-trip latency, so serial reads are latency-bound (8192 files *
        ~73 ms ≈ 10 min/shard).  Parallel reads saturate the volume's
        throughput instead of waiting on latency."""
        def _read(item):
            rel, p = item
            with open(p, "rb") as f:
                b = f.read()
            return rel, b
        with ThreadPoolExecutor(max_workers=32) as ex:
            rows = list(ex.map(_read, chunk))
        cols = {"image": [], "filename": [], "source": [], "size": []}
        for rel, b in rows:
            cols["image"].append(b)
            cols["filename"].append(rel)
            cols["source"].append(rel.split("/", 1)[0])
            cols["size"].append(len(b))
        return cols

    total_bytes = 0
    skipped = 0
    for i in range(n_shards):
        name = shard_name(i)
        if os.path.join(data_rel, name) in remote:
            print(f"[image-push] shard {name} already on hub — skipping")
            skipped += 1
            continue
        chunk = files[i * shard_rows:(i + 1) * shard_rows]
        t0 = time.time()
        cols = read_chunk(chunk)
        t_read = time.time() - t0
        table = pa.Table.from_pydict(cols, schema=schema)
        local = os.path.join(shard_dir, name)
        t1 = time.time()
        pq.write_table(table, local)
        t_write = time.time() - t1
        total_bytes += sum(len(b) for b in cols["image"])
        del cols, table
        t2 = time.time()
        api.upload_file(path_or_fileobj=local, path_in_repo=os.path.join(data_rel, name),
                        repo_id=repo_id, repo_type="dataset",
                        commit_message=f"shard {name}")
        t_upload = time.time() - t2
        print(f"[image-push] uploaded {name} ({len(chunk)} rows) "
              f"read={t_read:.0f}s write={t_write:.0f}s upload={t_upload:.0f}s")

    # dataset_info.json so the Hub viewer/server can validate + serve the dataset
    info = {
        "builder_name": "parquet",
        "config_name": "default",
        "dataset_name": "vision-adapter-images",
        "features": {
            "image": {"_type": "Binary"},
            "filename": {"dtype": "string", "_type": "Value"},
            "source": {"dtype": "string", "_type": "Value"},
            "size": {"dtype": "int64", "_type": "Value"},
        },
        "splits": {
            "train": {
                "name": "train",
                "num_examples": total,
                "num_bytes": total_bytes,
                "dataset_name": "vision-adapter-images",
            }
        },
    }
    if "dataset_info.json" not in remote:
        with open("/tmp/dataset_info.json", "w") as f:
            json.dump(info, f, indent=2, sort_keys=True)
        api.upload_file(path_or_fileobj="/tmp/dataset_info.json", path_in_repo="dataset_info.json",
                        repo_id=repo_id, repo_type="dataset")

    # dataset card for the image corpus
    card = f"""---
license: apache-2.0
size_categories:
- 10K<n<100K
annotations_creators:
- no-annotation
language_creators:
- machine-generated
language:
- en
multilinguality:
- monolingual
source_datasets:
- agentsea/wave-ui-25k
- showlab/ShowUI-desktop
- xlangai/aguvis-stage2
- HuggingFaceM4/the_cauldron
task_categories:
- image-to-text
- visual-question-answering
pretty_name: Vision Adapter Agentic + Cauldron Images
---

# Vision Adapter Image Corpus

Processed image corpus used to train the Vision-Adapter: 79,659 agentic UI images
("screenshots" + "multistep" subsets derived from wave-ui-25k, ShowUI-desktop, and
aguvis-stage2) plus full embeddable subsets of `HuggingFaceM4/the_cauldron` used for
general/reasoning/conversational fine-tuning.

Each image has been resized to ≤300k pixels and padded to 28-pixel multiples to match
MoonViT-V2 preprocessing.

## Fields

- `image`: raw PNG bytes (bytes column).
- `filename`: `source/basename` (e.g. `agentic/waveui_000123.png`).
- `source`: `agentic` or `cauldron`.
- `size`: bytes of the raw image.

## How to use

```python
from datasets import load_dataset

ds = load_dataset("{repo_ns}/vision-adapter-images")
print(ds)
```
"""
    if "README.md" not in remote:
        tmp_card = "/tmp/vision-adapter-images-README.md"
        with open(tmp_card, "w") as f:
            f.write(card)
        api.upload_file(path_or_fileobj=tmp_card,
                        path_in_repo="README.md",
                        repo_id=f"{repo_ns}/vision-adapter-images",
                        repo_type="dataset",
                        commit_message="Dataset card for image corpus")
    print(f"[image-push] success -> {repo_id}  ({total} rows, {skipped} shards already present)")


@app.function(image=_etl_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=60 * 30, memory="32GB", ephemeral_disk=524288)
def push_bench(repo_ns: str = "keypa", n_files: int = 8192):
    """Measure the three per-shard phases of push_image_corpus_to_hf separately
    so we know which one is actually slow: volume read, parquet write, HF upload.

    Uses the exact same calls/precursors as the real shard path."""
    import os, time
    from concurrent.futures import ThreadPoolExecutor
    vol.reload()

    files = []
    for grp in ("agentic", "cauldron"):
        grpdir = os.path.join(VOLUME_DIR, "images", grp)
        for fn in sorted(os.listdir(grpdir)):
            files.append((f"{grp}/{fn}", os.path.join(grpdir, fn)))
    files = files[:n_files]
    print(f"[push-bench] sample {len(files)} files")

    def _read(item):
        rel, p = item
        with open(p, "rb") as f:
            return rel, f.read()

    print("[push-bench] reading files in parallel (32 workers) ...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        rows = list(ex.map(_read, files))
    t_read = time.time() - t0
    byt = sum(len(b) for _, b in rows)
    print(f"[push-bench] READ:  {len(files)} files, {byt / 1e6:.0f} MB in {t_read:.0f}s "
          f"-> {byt / 1e6 / t_read:.1f} MB/s", flush=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        pa.field("image", pa.binary()),
        pa.field("filename", pa.string()),
        pa.field("source", pa.string()),
        pa.field("size", pa.int64()),
    ])
    cols = {"image": [], "filename": [], "source": [], "size": []}
    for rel, b in rows:
        cols["image"].append(b)
        cols["filename"].append(rel)
        cols["source"].append(rel.split("/", 1)[0])
        cols["size"].append(len(b))
    table = pa.Table.from_pydict(cols, schema=schema)
    local = "/tmp/bench.parquet"

    print("[push-bench] writing parquet ...", flush=True)
    t0 = time.time()
    pq.write_table(table, local)
    t_write = time.time() - t0
    print(f"[push-bench] WRITE: {byt / 1e6:.0f} MB in {t_write:.0f}s "
          f"-> {byt / 1e6 / t_write:.1f} MB/s", flush=True)

    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = f"{repo_ns}/push-bench-scratch"
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print("[push-bench] uploading real shard file to HF (same call as shard path) ...", flush=True)
    t0 = time.time()
    api.upload_file(path_or_fileobj=local, path_in_repo=os.path.basename(local),
                    repo_id=repo_id, repo_type="dataset", commit_message="shard bench")
    t_up = time.time() - t0
    mb = byt / 1e6
    print(f"[push-bench] UPLOAD: {mb:.0f} MB in {t_up:.0f}s -> {mb / t_up:.1f} MB/s", flush=True)

    print(f"[push-bench] per-shard estimate at these rates: read {t_read:.0f}s + "
          f"write {t_write:.0f}s + upload {t_up:.0f}s = {t_read + t_write + t_up:.0f}s", flush=True)


@app.function(image=_etl_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=60 * 60 * 2, memory="8GB")
def push_mix_manifest_to_hf(repo_ns: str = "keypa"):
    """Publish the curated 45/45/10 manifests plus cauldron metadata as a dataset.

    Uploads to `{repo_ns}/vision-adapter-manifests` (not `-data`), because the
    repo's purpose is the *recipes* (train split, held-out, cauldron metadata),
    not the underlying image bytes. Consuming this means you can rebuild the
    exact 45/45/10 mix and re-run precompute, without touching the Volume.
    """
    import os
    import shutil
    from huggingface_hub import HfApi, upload_file
    vol.reload()
    api = HfApi()

    repo_id = f"{repo_ns}/vision-adapter-manifests"
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # the three manifest and metadata files
    for rel in ["train_manifest.jsonl",
               "train_manifest_val.jsonl",
               "metadata/cauldron_manifest.jsonl"]:
        src = os.path.join(VOLUME_DIR, rel)
        if not os.path.exists(src):
            print(f"[push] SKIP missing {rel}")
            continue
        dst = os.path.join("/tmp", os.path.basename(rel))
        shutil.copy2(src, dst)
        upload_file(path_or_fileobj=dst,
                    path_in_repo=os.path.basename(rel),
                    repo_id=repo_id, repo_type="dataset")
        print(f"[push] {rel} -> {repo_id}")

    # HF dataset card for the manifest repo
    readme = f"""---
license: apache-2.0
size_categories:
- 100K<n<1M
task_categories:
- visual-question-answering
- image-to-text
pretty_name: Vision Adapter 45/45/10 SFT Manifest
---

# vision-adapter-manifests

The 45% agentic / 45% reasoning-doc / 10% conversational SFT mix (114,024 train
rows + 2,328 held-out validation rows) used to train the Vision-Adapter project.
Includes the full cauldron pull from which the mix was sampled.

The image corpus is a **separate** HF dataset repo
(`keypa/vision-adapter-images`).

## Contents

* `train_manifest.jsonl` — the actual train mixture (45% agentic / 45% doc / 10% conversational). Every row:
  `{{emb: "embeddings/<sha1>.pt", user, assistant, g: "agentic|doc|conv"}}`.
* `train_manifest_val.jsonl` — held-out validation split (never optimized).
* `cauldron_manifest.jsonl` — raw cauldron pull (≈1.9M rows) before the sampled recipe.
"""
    tmp = "/tmp/README.md"
    with open(tmp, "w") as f: f.write(readme)
    upload_file(path_or_fileobj=tmp, path_in_repo="README.md",
                repo_id=repo_id, repo_type="dataset", commit_message="Add README")
    print(f"[push] README -> {repo_id}")
    print(f"[push] DONE")


@app.function(image=_etl_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=60 * 60 * 8, memory="16GB",
              ephemeral_disk=524288)  # 512 GiB, matches etl()
def push_embeddings_to_hf(repo_ns: str = "keypa", shard_files: int = 10_000):
    """Publish the precomputed projector-input embeddings to a HF dataset repo.

    Every `.pt` cached by `precompute` under /data/embeddings/ is uploaded flat
    under `embeddings/`, so each remote path (`embeddings/<sha1>.pt`) matches the
    `emb:` field of train_manifest.jsonl exactly.  Uploads happen one commit per
    ~10k-file chunk (not per file), and chunks that already exist on the Hub are
    skipped, so an interrupted push resumes instead of restarting.
    """
    import os, glob, shutil
    from huggingface_hub import HfApi

    vol.reload()
    repo_id = f"{repo_ns}/vision-adapter-embeddings"
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    files = sorted(glob.glob(f"{EMB_DIR}/*.pt"))
    total = len(files)
    total_gb = sum(os.path.getsize(p) for p in files) / 1e9
    print(f"[emb-push] embeddings: {total}  {total_gb:.2f} GB")

    remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
    chunk_dir = "/tmp/emb_shards"
    os.makedirs(chunk_dir, exist_ok=True)

    n_uploaded = 0
    n_skipped = 0
    for s in range(0, total, shard_files):
        chunk = files[s:s + shard_files]
        all_present = all(f"embeddings/{os.path.basename(p)}" in remote
                          for p in chunk)
        if all_present:
            n_skipped += len(chunk)
            print(f"[emb-push] chunk {s // shard_files}: all {len(chunk)} present — skipping")
            continue
        # stage only the missing files into a fresh dir, then one commit per chunk
        stage = os.path.join(chunk_dir, f"chunk-{s // shard_files:03d}")
        if os.path.isdir(stage):
            shutil.rmtree(stage)
        os.makedirs(stage)
        pending = []
        for p in chunk:
            name = os.path.basename(p)
            if f"embeddings/{name}" in remote:
                continue
            shutil.copy2(p, os.path.join(stage, name))
            pending.append(name)
        if not pending:
            n_skipped += len(chunk)
            continue
        api.upload_folder(folder_path=stage, path_in_repo="embeddings",
                          repo_id=repo_id, repo_type="dataset",
                          commit_message=f"embeddings chunk {s // shard_files}")
        n_uploaded += len(pending)
        print(f"[emb-push] chunk {s // shard_files}: uploaded {len(pending)}")

    # dataset card for the embedding repo
    card = f"""---
license: apache-2.0
size_categories:
- 100K<n<1M
task_categories:
- visual-question-answering
- image-to-text
pretty_name: Vision Adapter Precomputed Embeddings
---

# vision-adapter-embeddings

Frozen MoonViT-V2 projector-input embeddings for the Vision-Adapter corpus,
precomputed once in Phase-3 (`precompute`) and cached as BF16 `.pt` tensors.

Each file is `torch.save` of a `[n_merged, 4096]` BF16 tensor — the flattened
4×1024 projector-input rows for one image.  `n_merged` is the number of merged
visual tokens (variable per image).

## Layout

* `embeddings/<sha1>.pt` — one file per image, flat, matching the `emb:` field
  of `keypa/vision-adapter-manifests` (`train_manifest.jsonl`):
  `emb: "embeddings/<sha1>.pt"` → `embeddings/<sha1>.pt` in this repo.
* The SHA1 is derived from the *volume-relative logical path*
  (`agentic/foo.png`, `cauldron/foo.png`), so this mirrors exactly what
  `modal_train.EmbSFT` loads.

## How to use

```python
import os, torch
from huggingface_hub import hf_hub_download
from datasets import load_dataset

man = load_dataset("keypa/vision-adapter-manifests", split="train")
row = man[0]
emb_path = hf_hub_download(
    "keypa/vision-adapter-embeddings", row["emb"], repo_type="dataset")
emb = torch.load(emb_path, map_location="cpu")   # [n_merged, 4096] bf16
print(emb.shape)
```

The image corpus lives separately at `keypa/vision-adapter-images`.
"""
    if "README.md" not in remote:
        tmp_card = "/tmp/vision-adapter-embeddings-README.md"
        with open(tmp_card, "w") as f:
            f.write(card)
        api.upload_file(path_or_fileobj=tmp_card, path_in_repo="README.md",
                        repo_id=repo_id, repo_type="dataset",
                        commit_message="Dataset card for embeddings")
    print(f"[emb-push] success -> {repo_id}  ({n_uploaded} uploaded, {n_skipped} already present)")


def bai_build_one(subset, bai):  # module-scope helper called from inside etl(); needs `os` at module top
    """Invoke the verified per-subset positional join and write into the Volume in parallel.

    Redirects the aguvis peek/zip dirs into the Modal Volume so the HF-sourced
    <name>-l1.json manifests and <name>.zip files survive across containers.
    """
    import concurrent.futures
    # redirect the /tmp/opencode dirs into the Modal volume so the manifests/zips persist
    if subset in ("aitw", "guiact-web-multi", "mind2web", "miniwob"):
        bai.AGUVIS_PEEK_DIR = os.path.join(VOLUME_DIR, "aguvis", "peek")
        bai.AGUVIS_ZIP_DIR = os.path.join(VOLUME_DIR, "aguvis", "zips")
        os.makedirs(bai.AGUVIS_PEEK_DIR, exist_ok=True)
        os.makedirs(bai.AGUVIS_ZIP_DIR, exist_ok=True)
    needed = bai.load_needed_indices()          # reads bai.MANIFEST
    idx_map = needed.get(subset, {})
    if not idx_map:  # nothing needed
        return
    datas = bai.build_subset(subset, idx_map)
    # Build all jobs *first* so we only decode+resize rows with real bytes
    jobs = []
    for i, _base in idx_map.items():
        ext = idx_map[i].rsplit(".", 1)[-1]
        out = os.path.join(bai.DEFAULT_OUT, f"{bai.sero_basename(subset, i, ext)}")
        if os.path.exists(out):
            continue
        jobs.append((os.path.basename(out), datas[i], ext, bai.DEFAULT_OUT))
    if not jobs:
        print(f"  [{subset}] all {len(idx_map)} images already present — skipping")
        return
    print(f"  [{subset}] writing {len(jobs)} images via 8-way pool...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(bai.process_one, jobs))
    print(f"  [{subset}] done")


DOC_SUBSETS = ["chartqa", "docvqa", "infographic_vqa", "screen2words", "websight",
               "ocrvqa", "textvqa", "plotqa", "ai2d", "scienceqa"]
CONV_SUBSETS = ["vqav2", "okvqa", "aokvqa", "visual7w"]

# Per-subset row caps: sample evenly from every subset for diversity without
# processing all 1.8M Cauldron rows.  Targets: 54k doc + 12k conv = 66k total.
# Doc: 54k / 10 subsets = 5400 each.  Conv: 12k / 4 subsets = 3000 each.
DOC_MAX_ROWS = 5400
CONV_MAX_ROWS = 3000


def cauldron_pull():
    """Download-then-open-local for each permissive cauldron subset.

    Parallelised: parquet shards download concurrently, rows are saved via a
    ThreadPoolExecutor (PNG encode is CPU-bound).  A checkpoint file tracks
    completed rec_ids so re-runs resume in seconds instead of hours."""
    import json, os, requests, threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import load_dataset
    from PIL import Image

    out_img = f"{IMG_DIR}/cauldron"; os.makedirs(out_img, exist_ok=True)
    manifest_path = f"{META_DIR}/cauldron_manifest.jsonl"
    done_path = f"{META_DIR}/cauldron_done.txt"
    os.makedirs(META_DIR, exist_ok=True)

    # --- checkpoint: rec_ids already fully processed ---
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            done = set(f.read().splitlines())
    print(f"[cauldron] checkpoint: {len(done)} rows already done")

    manifest_fh = open(manifest_path, "a")
    done_lock = threading.Lock()

    hdr = {"User-Agent": "vision-adapter/1.0"}
    cache_root = f"{VOLUME_DIR}/hf_cache/cauldron_parquet"
    os.makedirs(cache_root, exist_ok=True)

    N_DL = 6      # parallel parquet downloads
    N_SAVE = 12   # parallel PNG encodes (saturates CPU on 16-core)

    def save_one(rec_id, sub, imgs, texts):
        """Save images + append manifest for one row. Thread-safe."""
        from io import BytesIO
        paths = []
        for j, im in enumerate(imgs):
            p = f"{out_img}/{rec_id}-{j}.png"
            if not os.path.exists(p):
                # images can be bytes, dict{"bytes": ...}, or PIL depending on source
                if isinstance(im, dict):
                    im = Image.open(BytesIO(im["bytes"]))
                elif isinstance(im, bytes):
                    im = Image.open(BytesIO(im))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(p, optimize=False)
            paths.append(p)
        group = "doc" if sub in DOC_SUBSETS else "conv"
        lines = []
        for turn in texts:
            lines.append(json.dumps({"id": rec_id, "subset": sub, "group": group,
                                     "images": paths, "user": turn["user"],
                                     "assistant": turn["assistant"]}))
        with done_lock:
            manifest_fh.write("\n".join(lines) + "\n")
            manifest_fh.flush()
            with open(done_path, "a") as df:
                df.write(rec_id + "\n")
        sub

    def download_subset(sub, files):
        """Download all parquet shards for one subset using HF's parallel
        chunked downloader.  Much faster than sequential requests.get — uses
        multiple connections per file, CDN-aware.  Files are stored in a flat
        layout ({sub}__<filename>) so cache checks are instant."""
        from huggingface_hub import hf_hub_download
        from concurrent.futures import ThreadPoolExecutor as _TPE
        want = {f["rfilename"]: f.get("size", -1) for f in files}
        flat = {rel: os.path.join(cache_root, rel.replace("/", "__")) for rel in want}
        # check if all already cached (flat layout from a previous run)
        if all(os.path.exists(dest) and os.path.getsize(dest) == want[rel]
               for rel, dest in flat.items()):
            return [(dest, True) for dest in flat.values()]
        # download each shard in parallel; hf_hub_download is multi-connection
        def _dl(rel):
            dest = flat[rel]
            if os.path.exists(dest) and os.path.getsize(dest) == want[rel]:
                return dest, True
            hf_hub_download(
                repo_id="HuggingFaceM4/the_cauldron",
                repo_type="dataset",
                filename=rel,
                local_dir=cache_root,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download puts files in {local_dir}/{rel}; move to flat
            src = os.path.join(cache_root, rel)
            if src != dest and os.path.exists(src):
                os.rename(src, dest)
            return dest, False
        with _TPE(max_workers=N_DL) as ex:
            results = list(ex.map(_dl, want))
        # clean up any empty HF directory structure left behind
        hf_sub_dir = os.path.join(cache_root, "the_cauldron", sub)
        if os.path.isdir(hf_sub_dir) and not os.listdir(hf_sub_dir):
            os.rmdir(hf_sub_dir)
            parent = os.path.join(cache_root, "the_cauldron")
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        return results

    def subset_is_complete(sub, local_paths):
        """Return True if every row of this subset is already in the done set.

        Uses parquet metadata (num_rows only, no deserialization) to compute the
        expected rec_id range, then checks the checkpoint set.  This avoids the
        cost of re-reading + re-deserializing 100+ MB shards when a subset was
        fully processed in a previous run."""
        import pyarrow.parquet as pq
        try:
            total_rows = sum(pq.read_metadata(p).num_rows for p in local_paths)
        except Exception:
            return False
        # count how many done rec_ids belong to this subset
        n_done_sub = sum(1 for r in done if r.startswith(f"{sub}-"))
        return n_done_sub >= total_rows

    for sub in DOC_SUBSETS + CONV_SUBSETS:
        try:
            files = [s for s in requests.get(
                "https://huggingface.co/api/datasets/HuggingFaceM4/the_cauldron",
                params={"full": "true", "config": sub},
                headers=hdr, timeout=30).json().get("siblings", [])
                if f"the_cauldron/{sub}-" in s.get("rfilename", "")
                and s.get("rfilename", "").endswith(".parquet")]
            if not files:
                files = [s for s in requests.get(
                    "https://huggingface.co/api/datasets/HuggingFaceM4/the_cauldron",
                    params={"full": "true", "config": sub},
                    headers=hdr, timeout=30).json().get("siblings", [])
                    if sub in s.get("rfilename", "") and s.get("rfilename", "").endswith(".parquet")]
            # parallel download (HF snapshot_download, multi-connection per file)
            results = download_subset(sub, files)
            local_paths = [p for p, _ in results]
            total_mb = sum(os.path.getsize(p[0]) for p in results) / 1e6
            n_new = sum(1 for p in results if not p[1])
            print(f"[cauldron] {sub}: {len(results)} shards, {total_mb:.0f} MB total, {n_new} downloaded")
            # subset-level skip: if every row already done, no need to re-read parquet
            if subset_is_complete(sub, local_paths):
                print(f"[cauldron] {sub}: all rows already processed — skipping")
                # still delete parquet to free disk
                for p in local_paths:
                    if os.path.exists(p):
                        os.remove(p)
                continue
            import time, pyarrow.parquet as pq
            t0 = time.time()
            # read parquet directly with PyArrow (10-50x faster than HF datasets
            # for image columns — avoids per-row PIL deserialization overhead)
            table = pq.read_table(local_paths[0]) if len(local_paths) == 1 else pq.ParquetDataset(local_paths).read()
            images_col = table.column("images").to_pylist()
            texts_col = table.column("texts").to_pylist()
            n_rows = len(images_col)
            t_read = time.time() - t0
            print(f"[cauldron] {sub}: parquet read {t_read:.1f}s ({n_rows} rows)")
        except Exception as e:
            print(f"[cauldron] SKIP {sub}: {e}"); continue

        # parallel row processing
        n_done = 0
        # per-subset cap: sample evenly from every subset for diversity
        max_rows = DOC_MAX_ROWS if sub in DOC_SUBSETS else CONV_MAX_ROWS
        n_cap = min(n_rows, max_rows)
        if n_cap < n_rows:
            print(f"[cauldron] {sub}: capping {n_rows} → {n_cap} rows")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_SAVE) as ex:
            futures = {}
            for i in range(n_cap):
                rec_id = f"{sub}-{i:07d}"
                if rec_id in done:
                    n_done += 1
                    continue
                fut = ex.submit(save_one, rec_id, sub, images_col[i], texts_col[i])
                futures[fut] = i

            t_iter = time.time() - t0
            n_total = len(futures)
            print(f"[cauldron] {sub}: iterate {t_iter:.1f}s ({n_total} queued, {n_done} skipped)")
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[cauldron] row error {sub}#{i}: {e}")
                if i and i % 5000 == 0:
                    print(f"[cauldron] {sub}: {i} rows"); vol.commit()

        t_save = time.time() - t0
        rate = n_total / t_save if t_save > 0 else 0
        print(f"[cauldron] {sub}: save {t_save:.1f}s total ({n_total} rows, {rate:.0f} rows/s)")

        print(f"[cauldron] {sub}: done ({n_done} skipped, {len(futures)} processed)")
        vol.commit()

        # free disk: delete this subset's parquet cache (no longer needed)
        import shutil
        for p in local_paths:
            if os.path.exists(p):
                os.remove(p)
        vol.commit()
        print(f"[cauldron] {sub}: parquet cache deleted")

    manifest_fh.close()
    # count final manifest rows
    with open(manifest_path) as f:
        n_total = sum(1 for _ in f)
    print(f"[cauldron] manifest rows={n_total}")


# =====================================================================
# STAGE precompute
# =====================================================================
@app.function(image=_etl_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=60 * 60, memory="32GB")
def build_train_manifest():
    """Emit /data/train_manifest.jsonl in the exact schema modal_train.EmbSFT reads:
       {emb: <vol-relative .pt path>, user, assistant, g}.
       Mix: 45% agentic / 45% doc / 10% conversational by target counts."""
    import os, json, hashlib, random, duckdb
    random.seed(0)
    rows = []
    # agentic (~54k)
    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
    sero = con.execute(
        "SELECT image, conversations, source FROM read_parquet("
        "'https://huggingface.co/datasets/0xSero/glm-vision-sft-mix/resolve/"
        "refs%2Fconvert%2Fparquet/default/train/0000.parquet') "
        "WHERE source IN ('screenshots','multistep') LIMIT 54000").fetchall()
    for image, conv, src in sero:
        user = conv[0]["content"]
        ans = conv[1]["content"]
        # CRITICAL: hash the *logical* path (agentic/<file>) so it matches the
        # embedding filename convention used by `_emb_key` in precompute (Modal)
        # and precompute_colab (Colab). Never hash an absolute host path here.
        h = hashlib.sha1(f"agentic/{image}".encode()).hexdigest()[:20]
        rows.append({"emb": f"embeddings/{h}.pt", "user": user,
                     "assistant": ans, "g": "agentic"})
    # cauldron doc (~54k) + conv (~12k)
    with open(f"{META_DIR}/cauldron_manifest.jsonl") as f:
        cal = [json.loads(l) for l in f]
    doc = [m for m in cal if m["group"] == "doc"]
    cv  = [m for m in cal if m["group"] == "conv"]
    for group_rows, group_name, k in [(doc, "doc", 54000), (cv, "conv", 12000)]:
        for m in random.sample(group_rows, min(k, len(group_rows))):
            # m["images"][0] is the absolute path under IMG_DIR written by
            # cauldron_pull; strip everything up to '/images/' to recover the
            # logical path that _emb_key hashes on the producer side.
            img_abs = m["images"][0]
            rel = img_abs.split("/images/", 1)[-1]
            h = hashlib.sha1(rel.encode()).hexdigest()[:20]
            rows.append({"emb": f"embeddings/{h}.pt", "user": m["user"],
                         "assistant": m["assistant"], "g": group_name})
    random.shuffle(rows)
    # held-out val split (~2%, min 256) so the training loop has an honest
    # "is it grokking" probe that never sees optimisation updates.
    n_val = max(256, int(0.02 * len(rows)))
    val, train = rows[:n_val], rows[n_val:]
    with open(f"{VOLUME_DIR}/train_manifest.jsonl", "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open(f"{VOLUME_DIR}/train_manifest_val.jsonl", "w") as f:
        for r in val: f.write(json.dumps(r) + "\n")
    vol.commit()
    from collections import Counter
    print("[mix] train", Counter(r["g"] for r in train), "total", len(train))
    print("[mix] val  ", Counter(r["g"] for r in val), "total", len(val))

def _prefetch_packs(image_paths, batch, workers=8, ahead=1):
    """Generator of (chunk, pixel_values, grid_thws) with CPU decode+patchify
    running ahead of the GPU consumer.

    Decoding PNG + numpy patchify (collate_images) is the measured bottleneck of
    precompute — GPU saturates ~60% with serial decode because it idles waiting
    for the CPU.  Here a worker thread pool decodes images off-GPU while the
    accelerator computes the previous batch; `ahead` batches are prefetched, so
    util climbs toward the GPU's ceiling regardless of batch size.
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    from threading import Thread

    def _one(p):
        from PIL import Image
        from preprocess import process_image
        return process_image(Image.open(p).convert("RGB"))

    ex = ThreadPoolExecutor(max_workers=workers)
    q = Queue(maxsize=ahead)

    def _produce():
        try:
            for s in range(0, len(image_paths), batch):
                chunk = image_paths[s:s + batch]
                outs = list(ex.map(_one, chunk))
                pv = torch.cat([o["pixel_values"] for o in outs], dim=0)
                gt = torch.cat([o["grid_thws"] for o in outs], dim=0)
                q.put((chunk, pv, gt))
        finally:
            q.put(None)

    Thread(target=_produce, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            ex.shutdown(wait=True)
            return
        yield item


@app.function(image=_vit_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              gpu=GPU, timeout=60 * 60 * 12, memory="64GB")
def precompute(batch: int = _Const.BATCH, workers: int = 8, ahead: int = 2):
    """Frozen MoonViT-V2 over every image -> /data/embeddings/<sha1>.pt (BF16).

    `batch` is the micro-batch of images per forward; `workers` threads decode
    and patchify the *next* batch while the GPU computes the current one (the
    CPU decode is the real bottleneck, not the A100).  Bench showed img/s flat
    beyond B=32 at ~41 img/s with serial decode but VRAM still climbing, so
    batching past ~128 costs memory for nothing — prefer more `workers` instead.
    """
    import os, sys, glob, json, torch
    from huggingface_hub import hf_hub_download
    os.makedirs(EMB_DIR, exist_ok=True)
    for m in ["moonvit.py", "preprocess.py"]:
        p = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model", filename=m)
        with open(p) as f: open(f"/root/{m}", "w").write(f.read())
    sys.path.insert(0, "/root")
    from moonvit import load_moonvit_from_safetensors

    cfg = json.load(open(hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                                         filename="vision_config.json")))
    st  = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                          filename="moonvit_v2.safetensors")
    vit = load_moonvit_from_safetensors(st, cfg, device="cuda", dtype=torch.bfloat16)

    img_paths = sorted(glob.glob(f"{IMG_DIR}/agentic/*") + glob.glob(f"{IMG_DIR}/cauldron/*"))
    img_paths = [p for p in img_paths if not _already_done(p)]
    print(f"[precompute] total={len(img_paths)} new/todo={len(img_paths)} "
          f"batch={batch} workers={workers} ahead={ahead}")
    B = batch  # micro-batch of images per forward
    done = 0
    # saver pool: volume writes are network-backed (~ms each); do them off-loop so
    # the next forward starts before all N emb saves drain.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(16, max(4, workers))) as saver:
        with torch.no_grad():
            for chunk, pv, gt in _prefetch_packs(img_paths, B, workers=workers, ahead=ahead):
                merged = vit(pv.cuda(), gt.cuda())
                for p, emb in zip(chunk, merged):
                    # emb: [n_merged, 4, 1024]; cache the 4096-flattened projector input
                    flat = emb.reshape(emb.shape[0], -1).to(torch.bfloat16).cpu()
                    saver.submit(torch.save, flat, _emb_path(p))
                done += len(chunk)
                if done % (B * 50) == 0 or done == len(img_paths):
                    print(f"[precompute] {done}/{len(img_paths)}"); vol.commit()
    vol.commit()
    print(f"[precompute] DONE. wrote {done} embeddings to {EMB_DIR}")


@app.function(image=_vit_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              gpu=GPU, timeout=60 * 60 * 2, memory="64GB")
def precompute_bench(batch_sizes=(16, 32, 64, 128, 256, 512),
                     n_images: int = 512, n_iter: int = 3,
                     workers: int = 8, ahead: int = 2):
    """Measure GPU memory % and GPU utilization % for a range of micro-batch sizes.

    Same data path as `precompute` (decode PNG -> collate -> MoonViT-V2 forward),
    but runs `n_iter` passes over a fixed `n_images` sample per batch size and
    reports per size: wall img/s, peak GPU mem (GiB + % of the 80 GB A100), and
    GPU utilization % (sampled from nvidia-smi in a background thread). `workers`
    = decode threads, `ahead` = batches prefetched while the GPU runs the current
    one.  Use the numbers to pick the biggest batch that (a) fits comfortably and
    (b) drives util > 90% before spending A100-hours on the full 145k-image run.
    """
    import os, sys, glob, json, time, threading, subprocess, torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    os.makedirs(EMB_DIR, exist_ok=True)
    for m in ["moonvit.py", "preprocess.py"]:
        p = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model", filename=m)
        with open(p) as f: open(f"/root/{m}", "w").write(f.read())
    sys.path.insert(0, "/root")
    from moonvit import load_moonvit_from_safetensors
    from preprocess import collate_images  # serial-baseline path uses it

    cfg = json.load(open(hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                                         filename="vision_config.json")))
    st  = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                          filename="moonvit_v2.safetensors")
    vit = load_moonvit_from_safetensors(st, cfg, device="cuda", dtype=torch.bfloat16)
    n_param = sum(p.numel() for p in vit.parameters())
    try:
        import flash_attn  # noqa: F401
        flash_ok = True
    except Exception:
        flash_ok = False

    prop = torch.cuda.get_device_properties(0)
    total_gb = prop.total_memory / 2 ** 30

    vol.reload()
    img_paths = sorted(glob.glob(f"{IMG_DIR}/agentic/*") + glob.glob(f"{IMG_DIR}/cauldron/*"))
    img_paths = img_paths[:n_images]
    print(f"[bench] model={n_param/1e6:.0f}M params  flash_attn={'YES' if flash_ok else 'NO (slow fallback)'}  "
          f"card={prop.name} {total_gb:.0f} GiB  sample={len(img_paths)} images")

    n_stop = max(batch_sizes)
    n_use = (len(img_paths) // n_stop) * n_stop
    if n_use <= 0:
        raise RuntimeError(f"need >= {n_stop} images, found {len(img_paths)}")
    sample = img_paths[:n_use]

    # background nvidia-smi sampler for GPU % during each timed run
    def run_timed(B):
        res = {"util": [], "mem": []}
        stop = threading.Event()
        def _sample():
            while not stop.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=3).stdout.strip()
                    u, m = [int(x) for x in out.split(",")]
                    res["util"].append(u); res["mem"].append(m)
                except Exception:
                    pass
                time.sleep(0.05)
        def _loop(pf):
            with torch.no_grad():
                for _ in range(n_iter):
                    if pf:
                        # prefetch pipeline: decode batch k+1 while GPU runs batch k
                        for chunk, pv, gt in _prefetch_packs(sample, B, workers=workers,
                                                             ahead=ahead):
                            out = vit(pv.cuda(), gt.cuda())
                            for p, emb in zip(chunk, out):
                                emb.reshape(emb.shape[0], -1).to(torch.bfloat16).cpu()
                    else:
                        # serial baseline (old behavior)
                        for s in range(0, n_use, B):
                            chunk = sample[s:s + B]
                            ims = [Image.open(p).convert("RGB") for p in chunk]
                            pack = collate_images(ims)
                            out = vit(pack["pixel_values"].cuda(), pack["grid_thws"].cuda())
                            for p, emb in zip(chunk, out):
                                emb.reshape(emb.shape[0], -1).to(torch.bfloat16).cpu()
            torch.cuda.synchronize()
        thr = threading.Thread(target=_sample, daemon=True); thr.start()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _loop(pf=True)  # measure prefetch path for mem/util
        wall = time.time() - t0
        torch.cuda.synchronize()
        t_ser = time.time()
        _loop(pf=False)  # serial baseline for comparison
        wall_ser = time.time() - t_ser
        torch.cuda.synchronize()
        stop.set(); thr.join(timeout=5)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        avg_util = sum(res["util"]) / len(res["util"]) if res["util"] else float("nan")
        img_per_s = (n_use * n_iter) / wall
        img_per_s_ser = (n_use * n_iter) / wall_ser
        return wall, img_per_s, img_per_s_ser, peak, avg_util

    print(f"[bench] {'B':>5} {'pf img/s':>9} {'ser img/s':>9} {'peak GiB':>9} {'mem %':>7} {'gpu util %':>10}")
    results = []
    for B in batch_sizes:
        wall, imgps, imgps_ser, peak, util = run_timed(B)
        spd = imgps / imgps_ser if imgps_ser else float("nan")
        results.append((B, imgps, imgps_ser, peak, util))
        print(f"[bench] {B:>5} {imgps:>9.1f} {imgps_ser:>9.1f} {peak:>9.2f} {peak/total_gb*100:>6.1f}% {util:>9.1f}%  "
              f"({spd:.2f}x vs serial)")
    print("[bench] DONE. prefer `precompute(workers=N)` over raising batch; pick B with mem% < ~85.")
    return {"results": results, "flash_attn": flash_ok,
            "total_gb": total_gb, "n_images": n_use}


def _already_done(image_path):
    """Skip precompute for images whose embedding cache already exists and parses
    as a (n, 4096) tensor — the same size check the Colab variant performs."""
    import os, torch
    p = _emb_path(image_path)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return False
    try:
        d = torch.load(p, map_location="cpu", mmap=True)
        return isinstance(d, torch.Tensor) and d.dim() == 2 and d.shape[-1] == 4096
    except Exception:
        return False


def _emb_key(image_path):
    """Hash the VOLUME-RELATIVE logical path (agentic/foo.png, cauldron/foo.png)
    so Colab (/content/drive/..., raw name) and Modal (/data/images/...) produce
    the same embedding filename for the same logical image."""
    import os, hashlib
    rel = image_path.split("/images/", 1)[-1] if "/images/" in image_path else image_path
    return hashlib.sha1(rel.encode()).hexdigest()[:20] + ".pt"


def _emb_path(image_path):
    return f"{EMB_DIR}/{_emb_key(image_path)}"


# =====================================================================
# Training is handled by modal_train.py (single source of truth).
