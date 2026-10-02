#!/usr/bin/env python3
"""Generate from a trained projector, on real manifest rows.

The A/B that matters: same batch, same projector, only `position_ids`
differs. Before 2026-09-30 the legacy generation branch passed
inputs_embeds + attention_mask alone, so the image tokens sat on plain
arange positions the backbone was never trained on. A visible difference
between the two calls proves the fix is live.

Usage:
    python scripts/gen_check.py --ckpt emb_cache/checkpoints/projector_step200.pt
    python scripts/gen_check.py --ckpt ... --no-positions   # the "before" side
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=None,
                    help="local manifest; fetched from HF when absent")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--group", default="agentic")
    ap.add_argument("--n-vis", type=int, default=364)
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-positions", action="store_true",
                    help="reproduce the pre-fix behaviour (no position_ids)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vision_adapter.core import (
        build_projector,
        embeds_for,
        make_collate,
        train_position_ids,
        _resolve_positions_mode,
    )
    from vision_adapter.manifest import load_manifest

    ck = Path(args.ckpt)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    print(f"checkpoint {ck.name}  step={sd.get('step')}")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-2B", dtype=torch.bfloat16, device_map=args.device
    ).eval()
    # Qwen3.5's config IS the text config on some transformers versions and
    # wraps one on others — same getattr fallback train.py uses.
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
    rows, _ = load_manifest(mp)
    picks = [r for r in rows if r.get("g") == args.group][: args.n]
    if not picks:
        print(f"no rows in group {args.group!r}")
        return 1

    coll = make_collate(tok, tok.pad_token_id, max_len=4096, vision_dim=4096)

    def _strip_trailing_eos(batch, eos_id):
        """Cut the prefix before its first EOS.

        make_collate always terminates the training layout with EOS
        ([user][answer][EOS]); generating after an EOS yields an empty string
        every time, which looks exactly like a broken adapter. The production
        generator in colab_unsloth_test.py does the same cut.
        """
        import copy as _copy

        ids = batch["input_ids"][0]
        hits = (ids == eos_id).nonzero(as_tuple=False)
        if not len(hits):
            return batch, int(ids.shape[0])
        cut = int(hits[0].item())
        out = _copy.deepcopy(batch)
        out["input_ids"] = batch["input_ids"][:, :cut]
        out["attention_mask"] = batch["attention_mask"][:, :cut]
        out["labels"] = batch["labels"][:, :cut]
        return out, cut

    print(f"positions mode: {_resolve_positions_mode()}"
          f"{'  (FORCED OFF: reproducing the pre-fix path)' if args.no_positions else ''}")

    outputs = []
    for r in picks:
        # n_vis must agree with the row's grid_thw: the mismatch guard in
        # core.py rejects the pair otherwise. Derive it from the grid when the
        # manifest carries one, so a real row works rather than a fixture.
        grid = r.get("grid_thw")
        if grid:
            n_vis = int(grid[0]) * int(grid[1]) * int(grid[2]) // 4
        else:
            n_vis = args.n_vis
        # noise stands in for the MoonViT embedding: this checks the wiring and
        # the positions, not the image content
        vis = torch.randn(n_vis, 4096)
        batch = coll([{"vis": vis, "user": r["user"], "assistant": "",
                       "g": args.group, "grid_thw": grid}])
        gen_batch, cut = _strip_trailing_eos(batch, tok.eos_token_id)
        with torch.no_grad():
            inp = embeds_for(model, gen_batch, proj, args.device)
            kw = {}
            if not args.no_positions:
                kw["position_ids"] = train_position_ids(gen_batch).to(args.device)
            out = model.generate(
                inputs_embeds=inp["inputs_embeds"],
                attention_mask=inp["attention_mask"],
                max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tok.pad_token_id, **kw,
            )
        got = tok.decode(out[0][cut:], skip_special_tokens=True)
        outputs.append(got)
        print(f"\nUSER   : {r['user'][:90]}")
        print(f"EXPECT : {r['assistant']}")
        print(f"GOT    : {got!r}")

    if len(outputs) > 1:
        uniq = len(set(outputs))
        print(f"\n{uniq}/{len(outputs)} distinct outputs across prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
