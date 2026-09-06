#!/usr/bin/env python3
"""
Colab T4 test for Unsloth image with Qwen3.5-2B + MoonViT + projector checkpoints.
Run in Colab (Runtime -> T4 GPU):

!pip install -q huggingface_hub safetensors pillow torch transformers accelerate
!git clone https://github.com/keypaa/Vision-Adapter.git
%cd Vision-Adapter
# upload logs/checkpoints/projector_step200.pt (or pull from HF)
# then:
!python scripts/colab_unsloth_test.py --ckpt logs/checkpoints/projector_step200.pt --prompt "Describe the image."

For step10 vs step200 comparison, run twice.
"""
import argparse, io, json, requests, sys
from pathlib import Path
# Colab: repo not installed as package, add parent to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from vision_adapter.models.moonvit import load_moonvit_from_safetensors
from vision_adapter.models.preprocess import collate_images
from vision_adapter.core import HourglassProjector, make_collate, visual_inject

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/checkpoints/projector_step200.pt", help="projector checkpoint")
    ap.add_argument("--prompt", default="Describe the image.", help="user prompt")
    ap.add_argument("--url", default="https://unsloth.ai/cgi/image/unsloth_new_wb_logo_vnrA8AASj-jN5wy8UIYE-.png?format=raw", help="image URL (si --image non donné)")
    ap.add_argument("--image", default=None, help="chemin local image uploadée (ex: /content/image.png) — prioritaire sur --url")
    ap.add_argument("--max_new", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} cuda={torch.cuda.is_available()}")
    if device.type=="cuda":
        print(f"gpu {torch.cuda.get_device_name(0)} vram {torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB")

    # 1. image — local upload prioritaire sinon URL
    if args.image and Path(args.image).is_file():
        print(f"loading local image {args.image}")
        img = Image.open(args.image).convert("RGB")
    else:
        print(f"downloading {args.url}")
        r = requests.get(args.url, timeout=30)
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    print(f"image original {img.size}")
    # T4 14GB OOM for 2560x1344 -> 17664 patches (96x184) needs 13.9GB attention -> downscale for Colab
    if max(img.size) > 1024:
        scale = 1024 / max(img.size)
        new_w, new_h = int(img.size[0]*scale), int(img.size[1]*scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        print(f"downscaled for T4 {img.size} (max 1024 to avoid 13.9GB OOM, 17664 patches -> ~3000 patches)")
    else:
        print(f"image {img.size}")

    # 2. MoonViT
    print("loading MoonViT keypa/MoonViT-V2-Standalone")
    cfg = json.load(open(hf_hub_download(repo_id="keypa/MoonViT-V2-Standalone", repo_type="model", filename="vision_config.json")))
    st = hf_hub_download(repo_id="keypa/MoonViT-V2-Standalone", repo_type="model", filename="moonvit_v2.safetensors")
    vit = load_moonvit_from_safetensors(st, cfg, device=str(device), dtype=torch.bfloat16)
    print("MoonViT loaded")

    pack = collate_images([img])
    print(f"pack {pack['pixel_values'].shape} grid {pack['grid_thws'].tolist()}")
    with torch.no_grad():
        merged = vit(pack["pixel_values"].to(device).to(torch.bfloat16), pack["grid_thws"].to(device))
        emb = merged[0].reshape(merged[0].shape[0], -1)
        print(f"emb {emb.shape} n_vis={emb.shape[0]}")
        # n_vis for this 2000x maybe ~ 300 tokens
        vis = emb.to(torch.float32)  # train uses float

    # 3. Qwen + projector
    print("loading Qwen3.5-2B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=str(device))
    for p in model.parameters(): p.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg_llm = getattr(model.config, "text_config", model.config)
    llm_dim = int(cfg_llm.hidden_size)

    sd = torch.load(args.ckpt, map_location=str(device))
    state = sd.get("proj", sd)
    proj = HourglassProjector(4096, llm_dim).to(str(device), dtype=torch.bfloat16)
    proj.load_state_dict(state)
    print(f"projector {args.ckpt} loaded")

    coll = make_collate(tok, tok.pad_token_id, max_len=512, vision_dim=4096)
    items = [{"vis": vis, "user": args.prompt, "assistant": "", "g": "test"}]
    batch = coll(items)
    print(f"batch input_ids {batch['input_ids'].shape} n_vis {vis.shape[0]}")

    print(f"\nPrompt: {args.prompt!r}")
    print("Generating...")
    with visual_inject(batch, proj, model):
        out_ids = model.generate(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out_ids[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n=== {Path(args.ckpt).name} ===")
    print(f"Gen: {gen!r}")
    # also show loss if you want to compare step10 vs step200

if __name__ == "__main__":
    main()
