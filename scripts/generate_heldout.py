#!/usr/bin/env python3
"""Generate answers for measured, genuinely held-out agentic examples.

This deliberately uses only manifest rows mapped to emb_0000/emb_0001, which
are excluded from training. It streams the exact selected embedding keys and
requires each row's measured grid; it never substitutes a synthetic/noise image.

Example (Colab T4):
    VISION_ADAPTER_PROJECTOR=scaled python scripts/generate_heldout.py \
        --ckpt ./data/checkpoints/projector_p05_step4000.pt --n 3
"""
from __future__ import annotations

import argparse
import math
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.eval_heldout import HELDOUT_SHARDS

VISION_DIM = 4096
AGENTIC_MAX_VIS_TOKENS = 380


def _grid_n_vis(grid_thw) -> int:
    if grid_thw is None or len(grid_thw) != 3:
        raise ValueError(f"expected measured grid_thw with 3 values, got {grid_thw!r}")
    grid = [int(v) for v in grid_thw]
    if any(v <= 0 for v in grid):
        raise ValueError(f"grid_thw values must be positive, got {grid!r}")
    patches = math.prod(grid)
    if patches % 4:
        raise ValueError(f"grid_thw does not divide into 2x2-merged tokens: {grid!r}")
    return patches // 4


def select_agentic_heldout_rows(rows, index, n=3, seed=0,
                                max_n_vis=AGENTIC_MAX_VIS_TOKENS):
    """Choose deterministically among complete, in-regime held-out agentic rows.

    The returned rows carry private stream-location fields used to read those
    exact keys from the parquet shards. Missing measured grids are excluded,
    never replaced by inferred geometry. A present but inconsistent grid/index
    is treated as corrupt data and raises immediately.
    """
    candidates = []
    counts = {"not_heldout": 0, "not_agentic": 0, "missing_grid": 0,
              "over_token_cap": 0}
    for row in rows:
        loc = index.get(row.get("emb"))
        if loc is None or loc[0] not in HELDOUT_SHARDS:
            counts["not_heldout"] += 1
            continue
        if row.get("g") != "agentic":
            counts["not_agentic"] += 1
            continue
        grid = row.get("grid_thw")
        if grid is None:
            counts["missing_grid"] += 1
            continue
        n_vis = _grid_n_vis(grid)
        if len(loc) >= 3 and int(loc[2]) != n_vis:
            raise ValueError(
                f"measured grid/token mismatch for {row.get('emb')}: "
                f"grid gives {n_vis}, index gives {loc[2]}"
            )
        if n_vis > max_n_vis:
            counts["over_token_cap"] += 1
            continue
        if not row.get("user") or not row.get("assistant"):
            raise ValueError(f"held-out agentic row lacks prompt/answer: {row.get('emb')}")
        candidates.append({**row, "_stream_shard": loc[0],
                           "_stream_row": int(loc[1]), "_expected_n_vis": n_vis})

    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = candidates[:max(0, n)]
    return chosen, {**counts, "eligible": len(candidates), "selected": len(chosen)}


def plan_for_selected_rows(rows):
    """Build an exact EmbStreamDataset plan from rows already selected by key."""
    plan = {}
    for row in rows:
        shard = row["_stream_shard"]
        plan.setdefault(shard, []).append({**row, "_row": row["_stream_row"]})
    for shard_rows in plan.values():
        shard_rows.sort(key=lambda row: row["_row"])
    return plan


def validate_streamed_rows(selected, streamed):
    """Fail closed unless the stream returned every selected real embedding."""
    expected = {row["emb"]: row for row in selected}
    if len(expected) != len(selected):
        raise ValueError("selected manifest contains duplicate embedding keys")
    actual = {}
    for item in streamed:
        key = item.get("emb")
        if not key or key not in expected:
            raise RuntimeError(f"stream returned unexpected/missing embedding key: {key!r}")
        if key in actual:
            raise RuntimeError(f"stream returned duplicate embedding key: {key}")
        row = expected[key]
        if item.get("g") != "agentic" or item.get("user") != row["user"] \
                or item.get("assistant") != row["assistant"]:
            raise RuntimeError(f"streamed row content differs from manifest for {key}")
        grid = item.get("grid_thw")
        if grid is None or [int(v) for v in grid] != [int(v) for v in row["grid_thw"]]:
            raise RuntimeError(f"streamed row lost/changed measured grid for {key}")
        vis = item.get("vis")
        if not isinstance(vis, torch.Tensor) or tuple(vis.shape) != (
                row["_expected_n_vis"], VISION_DIM):
            shape = tuple(vis.shape) if isinstance(vis, torch.Tensor) else None
            raise RuntimeError(f"real embedding shape mismatch for {key}: {shape}")
        actual[key] = item
    missing = set(expected) - set(actual)
    if missing:
        raise RuntimeError(
            f"stream did not return {len(missing)} selected real embeddings: "
            f"{sorted(missing)[:5]}"
        )
    return [actual[row["emb"]] for row in selected]


def _load_backbone(device, dtype_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.type == "cuda":
        cc = torch.cuda.get_device_properties(device).major * 10 + torch.cuda.get_device_properties(device).minor
        if dtype_name == "auto":
            # T4 (7.5) has no native BF16; use FP16. Ampere+ uses BF16.
            dtype = torch.bfloat16 if cc >= 80 else torch.float16
        else:
            dtype = getattr(torch, dtype_name)
    else:
        dtype = torch.float32 if dtype_name == "auto" else getattr(torch, dtype_name)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-2B", dtype=dtype, low_cpu_mem_usage=True,
        device_map=str(device) if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    return tok, model, dtype


def _score_outputs(rows, outputs):
    verb = sum(got.startswith(row["assistant"].split("(")[0])
               for row, got in zip(rows, outputs))
    syntax = sum(("start_box" in got) == ("start_box" in row["assistant"])
                 for row, got in zip(rows, outputs))
    distances = []
    for row, got in zip(rows, outputs):
        got_xy = re.search(r"start_box=\[(-?\d+),\s*(-?\d+)\]", got)
        ref_xy = re.search(r"start_box=\[(-?\d+),\s*(-?\d+)\]", row["assistant"])
        if got_xy and ref_xy:
            distances.append(math.dist(
                (int(got_xy.group(1)), int(got_xy.group(2))),
                (int(ref_xy.group(1)), int(ref_xy.group(2))),
            ))
    print(f"\nformat/action: verb {verb}/{len(outputs)}, syntax {syntax}/{len(outputs)}")
    if distances:
        print(f"coordinate error: median {statistics.median(distances):.0f}px; "
              f"within 100px {sum(d < 100 for d in distances)}/{len(outputs)}")
    else:
        print("coordinate error: not scored (no start_box coordinates in both target and output)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="./data/checkpoints/projector_p05_step4000.pt")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-vis-tokens", type=int, default=AGENTIC_MAX_VIS_TOKENS)
    ap.add_argument("--gen-mode", choices=("greedy", "card"), default="card")
    ap.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = ap.parse_args()
    if args.n <= 0 or args.max_new <= 0 or args.max_seq_len <= args.max_new:
        ap.error("--n and --max-new must be positive; --max-seq-len must exceed --max-new")

    from vision_adapter.core import build_projector, embeds_for
    from vision_adapter.data.stream import (
        EmbStreamDataset, build_key_index, fetch_manifest,
    )
    from scripts.colab_unsloth_test import (
        build_generation_batch, manual_generate, strip_trailing_eos,
    )
    from vision_adapter.core import _resolve_positions_mode, train_position_ids

    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "cache"
    order = sorted(HELDOUT_SHARDS)
    print(f"[heldout-gen] measured manifest + index for {order}", flush=True)
    manifest_rows = fetch_manifest(cache_dir=str(cache_dir))
    index = build_key_index(order, cache_dir=str(cache_dir))
    selected, counts = select_agentic_heldout_rows(
        manifest_rows, index, n=args.n, seed=args.seed,
        max_n_vis=args.max_vis_tokens,
    )
    print(f"[heldout-gen] eligible/selected={counts['eligible']}/{counts['selected']} "
          f"excluded={{{', '.join(f'{k}:{v}' for k, v in counts.items() if k not in ('eligible', 'selected'))}}}",
          flush=True)
    if not selected:
        raise RuntimeError("no measured, held-out agentic rows in the requested visual-token regime")
    if len(selected) < args.n:
        print(f"[heldout-gen] only {len(selected)} of {args.n} requested rows are available", flush=True)

    plan = plan_for_selected_rows(selected)
    ds = EmbStreamDataset(
        plan, order, rg_cache_dir=str(cache_dir / "rg_cache"), vision_dim=VISION_DIM,
    )
    streamed = validate_streamed_rows(selected, list(ds))
    print(f"[heldout-gen] verified {len(streamed)} exact real embeddings; "
          "all grids measured and matched", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[heldout-gen] GPU {torch.cuda.get_device_name(device)} "
              f"({torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GiB)",
              flush=True)
    tok, model, dtype = _load_backbone(device, args.dtype)
    llm_dim = int(getattr(model.config, "text_config", model.config).hidden_size)
    variant = None
    import os
    variant = os.environ.get("VISION_ADAPTER_PROJECTOR") or None
    proj = build_projector(VISION_DIM, llm_dim, variant=variant)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"projector checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    proj.load_state_dict(state.get("proj", state))
    proj = proj.to(device=device, dtype=dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32).eval()
    print(f"[heldout-gen] checkpoint={ckpt_path} variant={variant or 'hourglass'} "
          f"step={state.get('step', 'unknown')} dtype={dtype} positions={_resolve_positions_mode()}",
          flush=True)

    outputs = []
    for row, item in zip(selected, streamed):
        grid = [int(v) for v in item["grid_thw"]]
        batch = build_generation_batch(
            tok, tok.pad_token_id, item["vis"], item["user"], grid,
            max_seq_len=args.max_seq_len, max_new_tokens=args.max_new,
        )
        gen_batch, cut = strip_trailing_eos(batch, tok.eos_token_id)
        inp = embeds_for(model, gen_batch, proj, str(device))
        positions = None
        if _resolve_positions_mode() == "mrope":
            positions = train_position_ids(gen_batch).to(device)
        new_ids = manual_generate(
            model, inp["inputs_embeds"], inp["attention_mask"],
            max_new_tokens=args.max_new, mode=args.gen_mode,
            eos_id=tok.eos_token_id, seed=args.seed, position_ids=positions,
        )
        got = tok.decode(new_ids, skip_special_tokens=True)
        outputs.append(got)
        print(f"\n[{row['emb']}] n_vis={item['vis'].shape[0]} grid_thw={grid}")
        print(f"USER   : {row['user']}")
        print(f"EXPECT : {row['assistant']}")
        print(f"GOT    : {got!r} (prefix tokens={cut})", flush=True)

    _score_outputs(selected, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
