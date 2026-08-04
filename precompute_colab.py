# %% [markdown]
# # MoonViT-V2 Embedding Precompute — Colab T4 (free tier, resumable)
#
# Precomputes frozen MoonViT-V2 merged embeddings (4096-dim per merged token) for the
# agentic + cauldron image corpus, so the Vision-Adapter projector can be trained on a
# small cached tensor set instead of running the ViT in the training hot loop.
#
# Hardware target: Google Colab Tesla T4 (15GB VRAM), free tier. MoonViT is ~0.8 GB in
# bf16; we use the remaining ~10GB for packed-patch activations by batching images.
# The job is fully resumable: embeddings are flushed to Google Drive every N steps and
# already-cached images are skipped, so a 4h session limit just restarts where it left.
#
# Setup (run once in a Colab cell before this script):
#   !git clone <your repo or upload these files>  # need moonvit.py, preprocess.py beside this file
#   !pip install safetensors pillow numpy huggingface_hub torch --quiet
#   from google.colab import drive; drive.mount('/content/drive')
#   # Put the image corpus under /content/drive/MyDrive/vision_adapter/images/{agentic,cauldron}
#   #   (export from your Modal Volume / local ETL), then point IMAGES_ROOT below at it.

# %%
import os, sys, time, json, glob, hashlib

# ------------------------------- config --------------------------------------
IMAGES_ROOT   = "/content/drive/MyDrive/vision_adapter/images"   # input corpus (png/jpg)
OUT_ROOT      = "/content/drive/MyDrive/vision_adapter/embeddings"  # .pt cache (Drive, persists)
MOONVIT_REPO  = "keypa/MoonViT-V2-Standalone"                     # pulls weights+cfg+code from HF
BF16          = True
BATCH_PATCHES = 240_000       # cap patches/step on T4 (~10GB headroom); tune up if you can
FLUSH_EVERY   = 2_000         # steps between Google Drive journal commits (log)
PROGRESS_EVERY= 250
DTYPE_STR     = "bf16"

sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")))  # beside moonvit.py / preprocess.py

# %%
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from moonvit import load_moonvit_from_safetensors
from preprocess import process_image, collate_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_outdir(path: str | None = None) -> str:
    p = path or OUT_ROOT
    os.makedirs(p, exist_ok=True)
    return p

def _emb_key(image_path: str) -> str:
    """Volume-relative hash key: strip everything up to and including the 'images/'
    marker so Modal (/data/images/...) and Colab Drive paths produce identical names
    for the same logical image (agentic/foo.png, cauldron/bar.png)."""
    rel = image_path.split("/images/", 1)[-1] if "/images/" in image_path else image_path
    return hashlib.sha1(rel.encode()).hexdigest()[:20] + ".pt"


def _emb_path(image_path: str) -> str:
    return os.path.join(OUT_ROOT, _emb_key(image_path))


def _already_done(image_path: str) -> bool:
    p = _emb_path(image_path)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return False
    try:  # cheap header validation — catches truncated writes from a killed cell
        d = torch.load(p, map_location="cpu", mmap=True)
        return isinstance(d, torch.Tensor) and d.dim() == 2 and d.shape[-1] == 4096
    except Exception:
        return False


def load_vit():
    cfg = json.load(open(hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                                         filename="vision_config.json")))
    st  = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                          filename="moonvit_v2.safetensors")
    vit = load_moonvit_from_safetensors(st, cfg, device=str(device),
                                        dtype=(torch.bfloat16 if BF16 else torch.float32))
    return vit


def enumerate_images(root: str):
    return sorted(glob.glob(os.path.join(root, "**", "*.*"), recursive=True))


def pack_batches(paths, batch_patch_cap):
    """Greedy-pack images into batches so total patches per step <= cap (T4 headroom)."""
    batches, cur, cur_p = [], [], 0
    cache = {}
    def patch_count(p):
        if p not in cache:
            with Image.open(p) as im:
                w, h = im.size
            import math
            scale = min(1.0, math.sqrt(65536 / max(1, (w // 14) * (h // 14))),
                               (512 * 14) / w, (512 * 14) / h)
            w2, h2 = min(int(w * scale), 7168), min(int(h * scale), 7168)
            pw = (w2 + 27) // 28 * 28 // 14
            ph = (h2 + 27) // 28 * 28 // 14
            cache[p] = max(1, pw * ph)   # raw patches (pre-merge)
        return cache[p]
    for p in paths:
        n = patch_count(p)
        if cur and cur_p + n > batch_patch_cap:
            batches.append(cur); cur, cur_p = [p], n
        else:
            cur.append(p); cur_p += n
    if cur: batches.append(cur)
    return batches


def run():
    _ensure_outdir()
    vit = load_vit()
    t0 = time.time()
    todos = [p for p in enumerate_images(IMAGES_ROOT) if not _already_done(p)]
    total = len(enumerate_images(IMAGES_ROOT))
    print(f"[precompute] corpus={total} images | remaining={len(todos)} "
          f"| cached_so_far={total - len(todos)}")
    if not todos:
        print("[precompute] nothing to do — cache is complete."); return
    batches = pack_batches(todos, BATCH_PATCHES)
    print(f"[precompute] packed into {len(batches)} batches (cap {BATCH_PATCHES} patches/step)")

    done = 0
    with torch.no_grad():
        for bi, chunk in enumerate(batches):
            ims = []
            for p in chunk:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            pack = collate_images(ims)
            merged = vit(pack["pixel_values"].to(device).to(vit.patch_embed.proj.weight.dtype
                                                            if BF16 else torch.float32),
                          pack["grid_thws"].to(device))
            for p, emb in zip(chunk, merged):
                flat = emb.reshape(emb.shape[0], -1)                 # [n_merged, 4096]
                out = flat.to(torch.bfloat16 if BF16 else torch.float32).cpu()
                torch.save(out, _emb_path(p))
            done += len(chunk)
            if done % PROGRESS_EVERY < len(chunk):
                used = torch.cuda.memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                rate = done / max(1e-6, time.time() - t0)
                eta = (len(todos) - done) / max(1e-6, rate) / 3600
                print(f"[precompute] {done}/{len(todos)}  "
                      f"gpu={used:.1f}GB peak={peak:.1f}GB  {rate:.1f} img/s  ETA {eta:.2f}h")
            if done and done % FLUSH_EVERY < len(chunk):
                with open(os.path.join(OUT_ROOT, "_journal.json"), "w") as f:
                    json.dump({"done": total - len(todos) + done, "total": total,
                               "ts": time.time()}, f)
                print("[precompute] journal flushed to Drive")
    print(f"[precompute] DONE. wrote {done} embeddings this session -> {OUT_ROOT}")


if __name__ == "__main__":
    run()
