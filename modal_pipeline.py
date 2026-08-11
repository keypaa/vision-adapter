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
_vit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "safetensors", "pillow", "numpy", "huggingface_hub", "accelerate")
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
    if n_existing >= 75_000:
        print(f"[etl] agentic corpus already on disk ({n_existing} images) — skipping agentic phase")
    else:
        needed = _hf_has("0xSero/glm-vision-sft-mix", "sero_manifest.parquet")
        mp_manifest = needed and "sero_manifest.parquet"
        if not mp_manifest:
            mp_manifest = _build_manifest_local()
        print("[etl] rebuilding agentic corpus from scratch …")
        needed = _load_needed_indices(limit=None, manifest_path=mp_manifest)
        for subset in _all_agentic_subsets():
            idx_map = needed.get(subset, {})
            if not idx_map:
                continue
            print(f"[etl] subset={subset} → building {len(idx_map)} images")
            bai = _load_agentic_bai()  # build_agentic_images.py
            bai_build_one(subset, bai)
            # commit after every subset so a preemption preserves all completed groups
            vol.commit()
        print("[etl] agentic corpus committed to Volume.")
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
              timeout=60 * 60, memory="16GB")
def push_image_corpus_to_hf(repo_ns: str = "keypa"):
    """Pack the processed agentic images into a single HF dataset.

    The corpus is written as parquet shards under `{repo_ns}/vision-adapter-images`.
    Each row is {image: bytes, filename: str, source: 'waveui'|'showui'|'aitw'|...}.
    Compatible with `datasets.load_dataset`. Images are lazily downloaded on demand.
    Use this if you want to skip per-file downloads of 79k source PNGs.
    """
    import os, pandas as pd
    from datasets import Dataset
    from huggingface_hub import HfApi

    vol.reload()
    api = HfApi()

    rows = []
    for grp in ("agentic", "cauldron"):
        grpdir = os.path.join(VOLUME_DIR, "images", grp)
        for fn in sorted(os.listdir(grpdir)):
            p = os.path.join(grpdir, fn)
            rows.append({
                "image": open(p, "rb").read(),
                "filename": f"{grp}/{fn}",
                "source": grp,
                "size": os.path.getsize(p),
            })
    df = pd.DataFrame(rows)
    print(f"[image-push] corpus rows: {len(df)}  total bytes: {df['size'].sum()/1e9:.2f} GB")

    os.makedirs(_datasets_tmp, exist_ok=True)
    ds = Dataset.from_pandas(df)
    ds.save_to_disk(_datasets_tmp)

    api.create_repo(
        f"{repo_ns}/vision-adapter-images",
        repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=f"{repo_ns}/vision-adapter-images",
        repo_type="dataset", folder_path=_datasets_tmp, path_in_repo="")

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
    tmp_card = "/tmp/vision-adapter-images-README.md"
    with open(tmp_card, "w") as f:
        f.write(card)
    api.upload_file(path_or_fileobj=tmp_card,
                    path_in_repo="README.md",
                    repo_id=f"{repo_ns}/vision-adapter-images",
                    repo_type="dataset",
                    commit_message="Dataset card for image corpus")
    print(f"[image-push] success -> {repo_ns}/vision-adapter-images")

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
  `{emb: "embeddings/<sha1>.pt", user, assistant, g: "agentic|doc|conv"}`.
* `train_manifest_val.jsonl` — held-out validation split (never optimized).
* `cauldron_manifest.jsonl` — raw cauldron pull (≈1.9M rows) before the sampled recipe.
"""
    tmp = "/tmp/README.md"
    with open(tmp, "w") as f: f.write(readme)
    upload_file(path_or_fileobj=tmp, path_in_repo="README.md",
                repo_id=repo_id, repo_type="dataset", commit_message="Add README")
    print(f"[push] README -> {repo_id}")
    print(f"[push] DONE")


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
            for p in results:
                tag = "cached" if p[1] else "downloaded"
                print(f"[cauldron] {sub}: {tag} {os.path.basename(p[0])} ({os.path.getsize(p[0])/1e6:.0f} MB)")
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
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_SAVE) as ex:
            futures = {}
            for i in range(n_rows):
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

@app.function(image=_vit_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              gpu=GPU, timeout=60 * 60 * 12, memory="64GB")
def precompute():
    import os, sys, glob, json, torch
    from huggingface_hub import hf_hub_download
    os.makedirs(EMB_DIR, exist_ok=True)
    for m in ["moonvit.py", "preprocess.py"]:
        p = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model", filename=m)
        with open(p) as f: open(f"/root/{m}", "w").write(f.read())
    sys.path.insert(0, "/root")
    from moonvit import load_moonvit_from_safetensors
    from preprocess import collate_images  # packs PIL list -> pixel_values, grid_thws
    from preprocess import process_image

    cfg = json.load(open(hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                                         filename="vision_config.json")))
    st  = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                          filename="moonvit_v2.safetensors")
    vit = load_moonvit_from_safetensors(st, cfg, device="cuda", dtype=torch.bfloat16)

    img_paths = sorted(glob.glob(f"{IMG_DIR}/agentic/*") + glob.glob(f"{IMG_DIR}/cauldron/*"))
    img_paths = [p for p in img_paths if not _already_done(p)]
    print(f"[precompute] total={len(img_paths)} new/todo={len(img_paths)}")
    B = _Const.BATCH  # micro-batch of images per forward
    done = 0
    with torch.no_grad():
        for s in range(0, len(img_paths), B):
            chunk = img_paths[s:s + B]
            outs = []
            from PIL import Image
            ims = [Image.open(p).convert("RGB") for p in chunk]
            pack = collate_images(ims)
            merged = vit(pack["pixel_values"].cuda(), pack["grid_thws"].cuda())
            for p, emb in zip(chunk, merged):
                # emb: [n_merged, 4, 1024]; cache the 4096-flattened projector input
                flat = emb.reshape(emb.shape[0], -1).to(torch.bfloat16).cpu()
                torch.save(flat, _emb_path(p))
            done += len(chunk)
            if done % (B * 50) == 0 or done == len(img_paths):
                print(f"[precompute] {done}/{len(img_paths)}"); vol.commit()
    vol.commit()
    print(f"[precompute] DONE. wrote {done} embeddings to {EMB_DIR}")


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
