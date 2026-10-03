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


def _real_embeddings(args, picks):
    """The stored MoonViT embedding per row, so the action verb is judgeable.

    Reuses visual_ablation's shard reader — one shard, one pass, only the
    wanted rows. Returns None when the embeddings cannot be reached, and the
    caller falls back to noise with a warning rather than failing silently.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from visual_ablation import _load_real_embeddings
    except Exception as exc:  # pragma: no cover - import shape only
        print(f"[real-emb] unavailable ({exc}); falling back to noise", flush=True)
        return None

    class _Args:
        shard = None
    try:
        out = _load_real_embeddings(_Args(), picks)
    except Exception as exc:
        print(f"[real-emb] could not stream ({exc}); falling back to noise", flush=True)
        return None
    if not out or all(t.numel() <= 1 for t in out):
        print("[real-emb] no embeddings loaded; falling back to noise", flush=True)
        return None
    return out


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
    ap.add_argument("--gen-mode", choices=("card", "greedy"), default="card",
                    help="card = the Qwen3.5 model-card sampling recipe (default); "
                         "greedy collapses to immediate EOS on this backbone")
    ap.add_argument("--no-real-emb", action="store_true",
                    help="fill the visual span with noise instead of the stored "
                         "MoonViT embedding (streams one shard; default is real)")
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

    from scripts.colab_unsloth_test import build_gen_kwargs

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
    real = _real_embeddings(args, picks) if not args.no_real_emb else None
    for i, r in enumerate(picks):
        # n_vis must agree with the row's grid_thw: the mismatch guard in
        # core.py rejects the pair otherwise. Derive it from the grid when the
        # manifest carries one, so a real row works rather than a fixture.
        grid = r.get("grid_thw")
        if grid:
            n_vis = int(grid[0]) * int(grid[1]) * int(grid[2]) // 4
        else:
            n_vis = args.n_vis
        if real is not None:
            vis = real[i]
        else:
            # Only with --no-real-emb: noise makes the action verb
            # unpredictable, so the model falls back to its narrative prior and
            # the output looks broken. That measured 10.15 nats of apparent
            # "format not learned" on a checkpoint that formats correctly.
            print("[WARN] noise visual span — the verb cannot be judged", flush=True)
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
                max_new_tokens=args.max_new,
                pad_token_id=tok.pad_token_id, **kw,
                # Greedy collapses to immediate EOS on Qwen3.5 — the model card
                # prescribes sampling for VL tasks. `card` is the prod default;
                # `greedy` is kept as an A/B on the sampler itself.
                **build_gen_kwargs(args.gen_mode),
            )
        # generate() with inputs_embeds returns ONLY the new tokens — there is
        # no prefix to skip. Slicing at `cut` (the prefix length) returned an
        # empty slice every time, which is why every prompt read ''.
        new_ids = out[0].tolist()
        got = tok.decode(new_ids, skip_special_tokens=True)
        if new_ids and all(i == tok.eos_token_id for i in new_ids[:3]):
            got = "<EOS immediately>"
        outputs.append(got)
        print(f"\nUSER   : {r['user'][:90]}")
        print(f"EXPECT : {r['assistant']}")
        print(f"GOT    : {got!r}")

    if len(outputs) > 1:
        uniq = len(set(outputs))
        print(f"\n{uniq}/{len(outputs)} distinct outputs across prompts")
    _report_scores(picks, outputs)
    return 0


def _report_scores(picks, outputs) -> None:
    """Correct-verb / syntax / grounding, so a `[500,300]` default reads as
    a failure rather than as a success."""
    import re
    import statistics

    verb = sum(o.startswith(p["assistant"].split("(")[0]) for p, o in zip(picks, outputs))
    syn = sum(("start_box" in o) == ("start_box" in p["assistant"])
              for p, o in zip(picks, outputs))
    dists = []
    for p, o in zip(picks, outputs):
        mg = re.search(r"start_box=\[(-?\d+),\s*(-?\d+)\]", o)
        me = re.search(r"start_box=\[(-?\d+),\s*(-?\d+)\]", p["assistant"])
        if mg and me:
            dists.append(((int(mg.group(1)) - int(me.group(1))) ** 2 +
                          (int(mg.group(2)) - int(me.group(2))) ** 2) ** 0.5)
    n = len(outputs)
    print(f"\nverb correct {verb}/{n}   syntax correct {syn}/{n}")
    if dists:
        print(f"coordinates: {sum(d < 100 for d in dists)}/{n} within 100 px, "
              f"median error {statistics.median(dists):.0f} px")


if __name__ == "__main__":
    raise SystemExit(main())
