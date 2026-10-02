#!/usr/bin/env python3
"""Does the loss actually depend on the image? Run it on a real checkpoint.

The historical run reached 111% of the Baseten grok reference (64,000
samples, 0.55 epoch) with a flat curve at loss ema 1.14. Before spending
GPU hours on another run, measure what the visual span is worth to the
loss: real vs zeroed vs shuffled, same batch, same projector.

A large `image_contribution` means the projector is using the image and
the problem is convergence. A small one means it is not, and neither
target_rms nor the sequence layout is the thing to fix first.

Usage:
    python scripts/visual_ablation.py --ckpt emb_cache/checkpoints/projector_step190.pt
    python scripts/visual_ablation.py --ckpt ... --group agentic --n 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--group", default="agentic")
    ap.add_argument("--n", type=int, default=8, help="rows to average over")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vision_adapter.core import build_projector, make_collate
    from vision_adapter.train import _val_rows_from_file, visual_ablation

    ck = Path(args.ckpt)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    print(f"checkpoint {ck.name}  step={sd.get('step')}")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-2B", dtype=torch.bfloat16, device_map=args.device
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    llm_cfg = getattr(model.config, "text_config", model.config)
    proj = build_projector(4096, int(llm_cfg.hidden_size))
    proj.load_state_dict(sd["proj"])
    proj = proj.to(args.device).float().eval()

    mp = args.manifest
    if not mp or not Path(mp).is_file():
        from huggingface_hub import hf_hub_download

        mp = hf_hub_download(
            "keypa/vision-adapter-manifests", "train_manifest_grids.jsonl",
            repo_type="dataset",
        )
    rows = _val_rows_from_file(mp)
    picks = [r for r in rows if r.get("g") == args.group][: args.n]
    if not picks:
        print(f"no rows in group {args.group!r}")
        return 1
    coll = make_collate(tok, tok.pad_token_id, max_len=4096, vision_dim=4096)

    reals, zeros, shufs, contribs = [], [], [], []
    for i, r in enumerate(picks):
        grid = r.get("grid_thw")
        n_vis = int(grid[0]) * int(grid[1]) * int(grid[2]) // 4 if grid else 364
        # noise, not the real embedding: this measures whether the loss PATH
        # reads the span, not whether the image content is informative
        vis = torch.randn(n_vis, 4096)
        batch = coll([{"vis": vis, "user": r["user"], "assistant": "",
                       "g": args.group, "grid_thw": grid}])
        out = visual_ablation(model, proj, batch, args.device)
        reals.append(out["real"])
        zeros.append(out["zero"])
        shufs.append(out["shuffled"])
        contribs.append(out["image_contribution"])
        print(f"row {i}: real={out['real']:.4f} zero={out['zero']:.4f} "
              f"shuffled={out['shuffled']:.4f} contrib={out['image_contribution']:+.4f}")

    def mean(xs):
        return sum(xs) / len(xs)

    mr, mz, ms, mc = mean(reals), mean(zeros), mean(shufs), mean(contribs)
    print(f"\n=== mean over {len(picks)} {args.group} rows ===")
    print(f"real              {mr:.4f}")
    print(f"zeroed visual     {mz:.4f}")
    print(f"shuffled visual   {ms:.4f}")
    print(f"image contribution (real - zero)  {mc:+.4f}  "
          f"({100 * mc / max(1e-9, mr):+.1f}% of the loss)")

    verdict = (
        "the projector IS using the image — the problem is convergence"
        if abs(mc) > 0.01
        else "the projector is NOT using the image — fix the injection "
             "(target_rms / layout) before training longer"
    )
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
