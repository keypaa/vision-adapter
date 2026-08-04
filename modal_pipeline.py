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
GROK_STEPS_TARGET = 1035                   # ~1 epoch @64 batch on 66k (Baseten)

_etl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("duckdb", "pyarrow", "pillow", "requests", "huggingface_hub", "datasets")
    .env({"HF_HOME": HF_CACHE})
)
_vit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "safetensors", "pillow", "numpy", "huggingface_hub", "accelerate")
    .env({"HF_HOME": HF_CACHE})
)

app = modal.App(APP_NAME)


# =====================================================================
# STAGE etl
# =====================================================================
@app.function(image=_etl_image, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=60 * 60 * 6, ephemeral_disk_request=250 * 1024)
def etl():
    import os, sys, json
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


def bai_build_one(subset, bai):
    """Invoke the verified per-subset positional join and write into the Volume."""
    needed = bai.load_needed_indices()          # reads bai.MANIFEST
    idx_map = needed.get(subset, {})
    if not idx_map:  # nothing needed
        return
    datas = bai.build_subset(subset, idx_map)
    for i, _base in idx_map.items():
        ext = idx_map[i].rsplit(".", 1)[-1]
        out = os.path.join(bai.DEFAULT_OUT, f"{bai.sero_basename(subset, i, ext)}")
        if os.path.exists(out):
            continue
        bai.process_one((os.path.basename(out), datas[i], ext, bai.DEFAULT_OUT))


DOC_SUBSETS = ["chartqa", "docvqa", "infographic_vqa", "screen2words", "websight",
               "ocrvqa", "textvqa", "plotqa", "ai2d", "scienceqa"]
CONV_SUBSETS = ["vqav2", "okvqa", "aokvqa", "visual7w"]


def cauldron_pull():
    """Stream cauldron permissive subsets; save as json rows + images in Volume."""
    import json, os
    from datasets import load_dataset
    from PIL import Image
    out_img = f"{IMG_DIR}/cauldron"; os.makedirs(out_img, exist_ok=True)
    manifest = []
    for sub in DOC_SUBSETS + CONV_SUBSETS:
        try:
            ds = load_dataset("HuggingFaceM4/the_cauldron", sub, split="train", streaming=True)
        except Exception as e:
            print(f"[cauldron] SKIP {sub}: {e}"); continue
        n = 0
        for i, row in enumerate(ds):
            try:
                imgs = row["images"]          # list[PIL]
                rec_id = f"{sub}-{i:07d}"
                paths = []
                for j, im in enumerate(imgs):
                    p = f"{out_img}/{rec_id}-{j}.png"
                    if not os.path.exists(p): im.save(p)
                    paths.append(p)
                for turn in row["texts"]:
                    manifest.append({"id": rec_id, "subset": sub,
                                     "group": ("doc" if sub in DOC_SUBSETS else "conv"),
                                     "images": paths,
                                     "user": turn["user"], "assistant": turn["assistant"]})
                n += 1
            except Exception as e:
                print(f"[cauldron] row error {sub}#{i}: {e}")
            if i and i % 5000 == 0:
                print(f"[cauldron] {sub}: {i} rows"); vol.commit()
        print(f"[cauldron] {sub}: {n} rows")
        vol.commit()
    with open(f"{META_DIR}/cauldron_manifest.jsonl", "w") as f:
        for m in manifest: f.write(json.dumps(m) + "\n")
    print(f"[cauldron] manifest rows={len(manifest)}")


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
    with open(f"{VOLUME_DIR}/train_manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    vol.commit()
    from collections import Counter
    print("[mix]", Counter(r["g"] for r in rows), "total", len(rows))

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

    cfg = json.load(open(hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                                         filename="vision_config.json")))
    st  = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                          filename="moonvit_v2.safetensors")
    vit = load_moonvit_from_safetensors(st, cfg, device="cuda", dtype=torch.bfloat16)

    img_paths = sorted(glob.glob(f"{IMG_DIR}/agentic/*") + glob.glob(f"{IMG_DIR}/cauldron/*"))
    print(f"[precompute] {len(img_paths)} images")
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
