#!/usr/bin/env python3
"""Can the projector overfit 16 fixed rows? The five-minute plumbing check.

Signature **S4** of the pre-registered outcome table (docs/DATA_PATH_REBUILD.md
4d): "the loss is pinned" would mean a BUG, not a science problem. The way to
tell the two apart in minutes instead of four GPU-hours is to train on a
handful of fixed rows and watch the loss fall. Memorising 16 rows is trivial
for a projector; if the loss will not move, then the collate, the position
ids, the LR or the projector scale are broken -- and a 4,000-step probe would
spend the same hours proving it.

This uses the SAME pieces the probe uses: the trainer's manifest, key index,
epoch plan, stream dataset, collate, projector init, optimizer and forward
step (``train_step_qwen``). So a pass here is evidence about the probe, not
about a toy.

    PASS  = the loss falls by >= 5x (typically 1.1 -> < 0.2): the plumbing
            works, and the probe's question is a science question.
    FAIL  = the loss stays put: the plumbing is broken. Do not spend GPU
            hours on the probe; find the bug first.

Usage:
    python scripts/overfit_test.py --data-dir ./data --rows 16 --steps 40
    python scripts/overfit_test.py --data-dir ./data --groups agentic --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = "Qwen/Qwen3.5-2B"


def verdict_for(curve: list[dict]) -> tuple[str, float, float, float]:
    """PASS if the projector can move; FAIL if it cannot.

    Two independent ways to pass, because either one is enough to clear the
    plumbing: the best loss gets small (a memorised batch), or it falls by a
    factor of five (the projector is learning, even if not yet memorising).
    """
    if not curve:
        return "FAIL", float("nan"), float("nan"), float("nan")
    first = curve[0]["loss"]
    best = min(c["loss"] for c in curve)
    last = curve[-1]["loss"]
    ratio = first / best if best > 0 else float("inf")
    ok = (best < 0.2) or (ratio >= 5.0)
    return ("PASS" if ok else "FAIL"), first, best, last


def load_rows(data_dir: Path, groups: str | None) -> list[dict]:
    """The same manifest preference the trainer uses (filtered > grids > plain)."""
    from vision_adapter.train import (
        FILTERED_MANIFEST_FILE,
        MEASURED_MANIFEST_FILE,
        PLAIN_MANIFEST_FILE,
    )
    from vision_adapter.manifest import load_manifest

    for name in (FILTERED_MANIFEST_FILE, MEASURED_MANIFEST_FILE, PLAIN_MANIFEST_FILE):
        p = data_dir / name
        if p.is_file():
            rows, _hdr = load_manifest(p)
            print(f"[overfit] manifest: {name} ({len(rows)} rows)")
            break
    else:
        raise SystemExit(f"[overfit] no train manifest in {data_dir}")

    if groups:
        want = {g.strip() for g in groups.split(",") if g.strip()}
        rows = [r for r in rows if r.get("g") in want]
        print(f"[overfit] groups={sorted(want)} -> {len(rows)} rows")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--rows", type=int, default=16, help="rows in the fixed batch")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4, help="the probe's LR")
    ap.add_argument("--groups", default="agentic",
                    help="manifest groups to draw from (the probe's slice)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the loss curve as JSON")
    args = ap.parse_args()

    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vision_adapter.core import (
        build_projector,
        make_collate,
        train_step_qwen,
    )
    from vision_adapter.train import resolve_dtype
    from vision_adapter.data.stream import (
        EmbStreamDataset,
        build_epoch_plan,
        build_key_index,
        list_shards,
    )

    data_dir = Path(args.data_dir)
    rows = load_rows(data_dir, args.groups)
    if len(rows) < args.rows:
        raise SystemExit(f"[overfit] only {len(rows)} rows available")

    # --- the same index/plan/dataset the trainer builds -------------------
    cache = data_dir / "cache"
    order = list_shards()
    print(f"[overfit] key index: {len(order)} shards", flush=True)
    index = build_key_index(order, cache_dir=str(cache))
    plan = build_epoch_plan(rows, index, sample_size=args.rows, seed=args.seed)
    n_planned = sum(len(v) for v in plan.values())
    print(f"[overfit] plan: {n_planned} rows from {len(plan)} shards", flush=True)

    # --- the same model/projector/optimizer the trainer uses --------------
    dev = args.device
    dtype = resolve_dtype(0, args.dtype) if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, low_cpu_mem_usage=True,
        device_map=dev if dev == "cuda" else None)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm_dim = int(getattr(model.config, "text_config", model.config).hidden_size)
    proj = build_projector(4096, llm_dim).to(dev, dtype=dtype)
    for p in proj.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, betas=(0.9, 0.95))
    collate = make_collate(tok, tok.pad_token_id, max_len=4096, vision_dim=4096)

    ds = EmbStreamDataset(plan, order, rg_cache_dir=str(cache / "rg_cache"),
                          vision_dim=4096)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.rows, drop_last=True,
                                         collate_fn=collate, num_workers=0)
    batch = next(iter(loader))
    print(f"[overfit] batch: L={batch['input_ids'].shape[1]} "
          f"vis={tuple(batch['vis'].shape)} groups={batch.get('g')}", flush=True)

    curve = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        out = train_step_qwen(model, proj, opt, batch, dev, clip=1.0)
        loss = out["loss"]
        curve.append({"step": step, "loss": loss, "gnorm": out["gnorm"]})
        print(f"[overfit] step {step:3d}/{args.steps}  loss={loss:.4f}  "
              f"gnorm={out['gnorm']:.3f}  {out['step_ms']:.0f}ms", flush=True)
        if not out["finite"]:
            print("[overfit] FAIL: the loss went non-finite")
            break

    first = curve[0]["loss"] if curve else float("nan")
    last = curve[-1]["loss"] if curve else float("nan")
    best = min((c["loss"] for c in curve), default=float("nan"))
    verdict, first, best, last = verdict_for(curve)
    print(f"\n[overfit] first={first:.4f} best={best:.4f} last={last:.4f} "
          f"({time.time() - t0:.0f}s) -> {verdict}")
    if verdict == "PASS":
        print("[overfit] the projector CAN move: the collate, the position ids, "
              "the LR and the projector scale are working. The probe's question "
              "is a science question, not a plumbing one.")
    else:
        print("[overfit] the projector CANNOT move on a fixed batch: the "
              "plumbing is broken. Do not spend GPU hours on the probe -- find "
              "the bug first (collate / position ids / LR / projector scale).")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"verdict": verdict, "first": first, "best": best, "last": last,
             "curve": curve}, indent=1))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
