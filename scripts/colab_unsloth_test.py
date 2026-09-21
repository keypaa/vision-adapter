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
import argparse
import io
import json
import requests
import sys
from pathlib import Path
# Colab: repo not installed as package, add parent to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vision_adapter.models.moonvit import load_moonvit_from_safetensors
from vision_adapter.models.preprocess import collate_images
from vision_adapter.core import build_projector, make_collate, embeds_for

def build_gen_kwargs(mode):
    """Sampling params per Qwen3.5 model card (non-thinking VL recipe).

    Greedy (do_sample=False) collapses to immediate EOS on this
    post-trained model; the card prescribes sampling for VL tasks.
    """
    if mode == "greedy":
        return {"do_sample": False}
    if mode == "card":
        return {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
    raise ValueError(f"unknown gen mode {mode!r} (expected greedy|card)")


def resolve_qwen_class(native: bool):
    """Backbone class per prefix mode: the multimodal forward (mm kwargs,
    mRoPE, masked scatter) only exists on Qwen3_5ForConditionalGeneration;
    the causal class exposes a text-only decoder. Concrete class import:
    the generic auto alias does not exist on all transformers versions."""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    return Qwen3_5ForConditionalGeneration if native else AutoModelForCausalLM


def strip_trailing_eos(batch, eos_id):
    """Cut input_ids/attention_mask before the first EOS for generation.

    make_collate always terminates the prefix with EOS (train layout
    [user][answer][EOS]); generating after an EOS yields empty output.
    Other batch keys pass through untouched. No EOS → full length.
    """
    ids = batch["input_ids"][0]
    hits = (ids == eos_id).nonzero(as_tuple=False)
    if len(hits):
        cut = int(hits[0].item())
    else:
        cut = int(batch["attention_mask"].sum().item())
    gen_batch = {
        k: (v[:, :cut] if k in ("input_ids", "attention_mask") else v)
        for k, v in batch.items()
    }
    return gen_batch, cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/checkpoints/projector_step200.pt", help="projector checkpoint")
    ap.add_argument("--prompt", default="Describe the image.", help="user prompt")
    ap.add_argument("--url", default="https://unsloth.ai/cgi/image/unsloth_new_wb_logo_vnrA8AASj-jN5wy8UIYE-.png?format=raw", help="image URL (si --image non donné)")
    ap.add_argument("--image", default=None, help="chemin local image uploadée (ex: /content/image.png) — prioritaire sur --url")
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--gen-mode", choices=("greedy", "card"), default="card")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug-ids", action="store_true", help="print raw generated ids + text-only control")
    ap.add_argument("--prefix-mode", choices=("legacy", "native"), default="legacy",
                    help="legacy: raw embeds_for splice; native: creator protocol (placeholders+mm+mRoPE)")
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

    # 3. Qwen + projector (native prefix needs the conditional class:
    # only it implements the multimodal forward with mm kwargs + mRoPE)
    qwen_cls = resolve_qwen_class(args.prefix_mode == "native")
    print(f"loading Qwen3.5-2B ({qwen_cls.__name__})")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = qwen_cls.from_pretrained("Qwen/Qwen3.5-2B", dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=str(device))
    for p in model.parameters():
        p.requires_grad_(False)
    # Eval harness: ckpt recompute corrupts generate (KV-cache), train() enables dropout.
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    cfg_llm = getattr(model.config, "text_config", model.config)
    llm_dim = int(cfg_llm.hidden_size)

    # Full-state step ckpts embed optimizer/RNG state (numpy) — trusted source (own run), same as train.py resume.
    sd = torch.load(args.ckpt, map_location=str(device), weights_only=False)
    state = sd.get("proj", sd)
    proj = build_projector(4096, llm_dim).to(str(device), dtype=torch.bfloat16)
    proj.load_state_dict(state)
    print(f"projector {args.ckpt} loaded")

    coll = make_collate(tok, tok.pad_token_id, max_len=512, vision_dim=4096)
    items = [{"vis": vis, "user": args.prompt, "assistant": "", "g": "test"}]
    batch = coll(items)
    print(f"batch input_ids {batch['input_ids'].shape} n_vis {vis.shape[0]}")

    print(f"\nPrompt: {args.prompt!r}")
    print("Generating (Qwen embeds_for, not DeepSeek hook)...")
    extra_kwargs: dict = {}
    if args.prefix_mode == "native":
        from scripts.native_prefix import build_native_generate_inputs, grid_for_nvis

        grid = torch.stack([grid_for_nvis(int(n)) for n in batch["n_vis"].tolist()])
        print(f"native grid (synthetic, N==n_vis): {grid.tolist()}")
        nin = build_native_generate_inputs(model, proj, batch, tok, grid, str(device))
        inp = {"inputs_embeds": nin["inputs_embeds"], "attention_mask": nin["attention_mask"]}
        extra_kwargs = {"mm_token_type_ids": nin["mm_token_type_ids"],
                        "image_grid_thw": nin["image_grid_thw"], "position_ids": nin["position_ids"]}
        cut = int(nin["attention_mask"][0].sum().item())
        print(f"native prefix live={cut} mm1={(nin['mm_token_type_ids'] == 1).sum().item()} pos={tuple(nin['position_ids'].shape)}")
        vis_mask = (nin["attention_mask"].bool() & (nin["mm_token_type_ids"] == 1))
    else:
        gen_batch, cut = strip_trailing_eos(batch, tok.eos_token_id)
        print(f"prefix cut at {cut} (was {batch['input_ids'].shape[1]}) — EOS-terminated train layout would generate empty")
        inp = embeds_for(model, gen_batch, proj, str(device))
        vis_mask = None
    torch.manual_seed(args.seed)
    out_ids = model.generate(inputs_embeds=inp["inputs_embeds"], attention_mask=inp["attention_mask"], max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id, **extra_kwargs, **build_gen_kwargs(args.gen_mode))
    gen = tok.decode(out_ids[0][cut:], skip_special_tokens=True)
    print(f"\n=== {Path(args.ckpt).name} ===")
    print(f"Gen: {gen!r}")
    if args.debug_ids:
        new_ids = out_ids[0][cut:].tolist()
        print(f"DEBUG new_tokens={len(new_ids)} ids={new_ids[:20]}")
        print(f"DEBUG embeds dtype={inp['inputs_embeds'].dtype} shape={tuple(inp['inputs_embeds'].shape)} "
              f"has_nan={bool(torch.isnan(inp['inputs_embeds']).any())} "
              f"has_inf={bool(torch.isinf(inp['inputs_embeds']).any())} "
              f"mask_sum={int(inp['attention_mask'].sum())}")
        if vis_mask is not None:
            vv = inp["inputs_embeds"][vis_mask].float()
            tt = inp["inputs_embeds"][inp["attention_mask"].bool() & ~vis_mask].float()
            print(f"DEBUG vis_embed mean={vv.mean():.4f} absmax={vv.abs().max():.4f} rms={vv.pow(2).mean().sqrt():.4f} "
                  f"vs text mean={tt.mean():.4f} absmax={tt.abs().max():.4f} rms={tt.pow(2).mean().sqrt():.4f}")
        # Run B: same prefix, no KV-cache (isolates cache/MTP interaction with inputs_embeds).
        torch.manual_seed(args.seed)
        out_nocache = model.generate(inputs_embeds=inp["inputs_embeds"], attention_mask=inp["attention_mask"],
                                     max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id,
                                     use_cache=False, **build_gen_kwargs(args.gen_mode))
        nc_new = out_nocache[0][cut:].tolist()
        print(f"DEBUG nocache new_tokens={len(nc_new)} ids={nc_new[:20]}")
        print(f"DEBUG nocache Gen: {tok.decode(nc_new, skip_special_tokens=True)!r}")
        # Text-only control: same model+sampling, no visual injection.
        # Empty here too => generate path broken; non-empty => image conditioning issue.
        t = tok(args.prompt, return_tensors="pt").to(str(device))
        torch.manual_seed(args.seed)
        t_out = model.generate(**t, max_new_tokens=32, pad_token_id=tok.pad_token_id, **build_gen_kwargs(args.gen_mode))
        t_new = t_out[0][t["input_ids"].shape[1]:].tolist()
        print(f"DEBUG text-only new_tokens={len(t_new)} ids={t_new[:20]}")
        print(f"DEBUG text-only Gen: {tok.decode(t_new, skip_special_tokens=True)!r}")
        # Embeds control: same text-only ids routed through inputs_embeds.
        # Non-empty => generate-from-embeds works, visual content is suspect;
        # empty too => inputs_embeds path broken on this model, end of track.
        with torch.no_grad():
            t_emb = model.get_input_embeddings()(t["input_ids"])
        torch.manual_seed(args.seed)
        e_out = model.generate(inputs_embeds=t_emb, attention_mask=t["attention_mask"],
                               max_new_tokens=32, pad_token_id=tok.pad_token_id,
                               **build_gen_kwargs(args.gen_mode))
        e_new = e_out[0][t["input_ids"].shape[1]:].tolist()
        print(f"DEBUG embeds-control new_tokens={len(e_new)} ids={e_new[:20]}")
        print(f"DEBUG embeds-control Gen: {tok.decode(e_new, skip_special_tokens=True)!r}")

if __name__ == "__main__":
    main()
