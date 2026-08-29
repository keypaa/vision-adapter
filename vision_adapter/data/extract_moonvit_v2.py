#!/usr/bin/env python3
"""Phase 2: Extract MoonViT-V2 (vision tower) + Kimi mm_projector from Kimi-K3.

Kimi-K3 is 2.8T params / 96 shards / ~1.5TB BF16-packed. We CANNOT download all of it.
The weight_map shows the vision subsystem lives ENTIRELY in the last 2 shards:
    model-00095-of-000096.safetensors   (mm_projector.*)
    model-00096-of-000096.safetensors   (vision_tower.*)
We stream-download only those 2 shards (~24GB), carve out the vision tensors, and
save standalone safetensors that we push to the Hub so this is a one-time cost.

Outputs (pushed to keypa/MoonViT-V2-Standalone):
    moonvit_v2.safetensors        vision encoder only, ~401M params, BF16
    kimi_mm_projector.safetensors Kimi's PatchMerger+proj (4096->7168), ~29M params
    vision_config.json            Kimi vision_config verbatim
"""
import os, json, sys, requests
from huggingface_hub import hf_hub_download, create_repo, upload_file
from safetensors import safe_open
from safetensors.torch import save_file
import torch

REPO = "moonshotai/Kimi-K3"
OUT_REPO = os.environ.get("MOONVIT_REPO", "keypa/MoonViT-V2-Standalone")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/opencode/moonvit_out")
VIS_ROOTS = ("vision_tower",)        # encoder
PROJ_ROOTS = ("mm_projector",)       # Kimi's own projector (reference target)
INDEX_URL = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"


def _tok(required=False):
    # env vars first; else fall back to the CLI-saved token (~/.cache/huggingface/token)
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t:
        try:
            from huggingface_hub import get_token
            t = get_token()
        except Exception:
            t = None
    if not t and required:
        sys.exit("No HF token found (env or `hf auth login`). Required for hub upload.")
    return t  # None is fine for downloads: Kimi-K3 shards are public/ungated.


def fetch_index():
    r = requests.get(INDEX_URL, timeout=120)  # public; no auth header needed
    r.raise_for_status()
    d = r.json()
    return d["weight_map"]


def select_keys(wm, roots):
    keys = sorted(k for k in wm if k.split(".")[0] in roots)
    shards = sorted({wm[k] for k in keys})
    return keys, shards


def carve(shards, keep_keys):
    """Load each shard once, pull only keep_keys. Returns state_dict."""
    keep = set(keep_keys)
    state = {}
    for sh in shards:
        path = hf_hub_download(REPO, sh)  # anonymous OK: public repo
        print(f"  carving {sh}  ({os.path.getsize(path)/1e9:.2f} GB on disk)")
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                if k in keep:
                    state[k] = f.get_tensor(k)
    return state


def n_param(sd):
    return sum(v.numel() for v in sd.values())


def main():  # keys retain their canonical Kimi prefixes (vision_tower.* / mm_projector.*); loader remap handles the namespace

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[1/5] fetching weight_map from {REPO} ...")
    wm = fetch_index()

    vis_keys, vis_shards = select_keys(wm, VIS_ROOTS)
    proj_keys, proj_shards = select_keys(wm, PROJ_ROOTS)
    all_shards = sorted(set(vis_shards) | set(proj_shards))
    print(f"      vision tensors={len(vis_keys)} in {vis_shards}")
    print(f"      projector tensors={len(proj_keys)} in {proj_shards}")
    print(f"      -> need {len(all_shards)} shards total: {all_shards}")

    print("[2/5] downloading + carving vision tower ...")
    vis = carve(all_shards, vis_keys)          # both shards; vision_tower is in 96
    print(f"      extracted {len(vis)} tensors = {n_param(vis)/1e6:.1f}M params")

    print("[3/5] carving Kimi mm_projector (reference) ...")
    proj = carve(all_shards, proj_keys)        # projector is in 95
    print(f"      extracted {len(proj)} tensors = {n_param(proj)/1e6:.1f}M params")

    # ---- sanity expectations
    assert 350e6 < n_param(vis) < 450e6, f"vision param count off: {n_param(vis)/1e6:.1f}M"
    assert all(v.dtype == torch.bfloat16 for v in vis.values()), "expected BF16 vision weights"

    print("[4/5] saving standalone safetensors ...")
    vis_path = os.path.join(OUT_DIR, "moonvit_v2.safetensors")
    proj_path = os.path.join(OUT_DIR, "kimi_mm_projector.safetensors")
    save_file(vis, vis_path, metadata={"format": "pt", "source": REPO})
    save_file(proj, proj_path, metadata={"format": "pt", "source": REPO})

    # vision_config verbatim + a standalone card
    cfg = json.loads(requests.get(
        f"https://huggingface.co/{REPO}/raw/main/config.json").text)["vision_config"]
    cfg_path = os.path.join(OUT_DIR, "vision_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"      wrote {vis_path}")
    print(f"      wrote {proj_path}")
    print(f"      wrote {cfg_path}")

    if os.environ.get("PUSH", "0") == "1":
        tok = _tok(required=True)
        print(f"[5/5] pushing to hub repo {OUT_REPO} ...")
        create_repo(OUT_REPO, exist_ok=True, token=tok)
        for p, name in [(vis_path, "moonvit_v2.safetensors"),
                        (proj_path, "kimi_mm_projector.safetensors"),
                        (cfg_path, "vision_config.json")]:
            upload_file(path_or_fileobj=p, path_in_repo=name,
                        repo_id=OUT_REPO, token=tok)
            print(f"      uploaded {name}")
    else:
        print("[5/5] PUSH!=1, skipping hub upload (set PUSH=1 to publish).")

    print("DONE.")
    return vis_path, proj_path, cfg_path


if __name__ == "__main__":
    main()
