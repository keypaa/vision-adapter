#!/usr/bin/env python3
"""Held-out eval gate for the probe (Phase 3b grok check).

Compares a final projector ckpt against a baseline ckpt on rows from
shards NEVER trained on (emb_0000/emb_0001 are excluded from every
training plan), plus leaves generation comparison to
scripts/colab_unsloth_test.py.

Usage (Molab GPU):
    export HF_TOKEN=...
    python scripts/eval_heldout.py --n 60 --data-dir ./data
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

HELDOUT_SHARDS = {"data/emb_0000.parquet", "data/emb_0001.parquet"}


def select_heldout_rows(rows, index, heldout_shards=HELDOUT_SHARDS, n=60, seed=0):
    """Deterministic subset of manifest rows whose embedding lives in a held-out shard."""
    cands = [r for r in rows if index.get(r.get("emb"), (None,))[0] in heldout_shards]
    rng = random.Random(seed)
    rng.shuffle(cands)
    return cands[: max(0, n)]


def forward_loss(model, proj, batch, device):
    """Selective lm_head loss, no backward/step. Mirrors train_step_qwen fwd
    exactly (same embeds_for splice AND same mRoPE train_position_ids)."""
    import torch.nn.functional as F

    from vision_adapter.core import embeds_for, train_position_ids

    prev = model.training
    model.eval()
    try:
        with torch.no_grad():
            inp = embeds_for(model, batch, proj, device)
            pos = train_position_ids(batch).to(inp["inputs_embeds"].device)
            out_dtype = next(model.parameters()).dtype
            amp_dtype = out_dtype if device == "cuda" else None
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model.model(inputs_embeds=inp["inputs_embeds"], attention_mask=inp["attention_mask"],
                                  position_ids=pos)
                hidden = out.last_hidden_state
                shift_labels = inp["labels"][:, 1:]
                mask = shift_labels != -100
                pos = mask.nonzero(as_tuple=False)
                h_sel = hidden[:, :-1][pos[:, 0], pos[:, 1]]
                y_sel = shift_labels[pos[:, 0], pos[:, 1]]
                logits_sel = model.lm_head(h_sel).float()
            loss = F.cross_entropy(logits_sel, y_sel)
            # tokens follows the train contract: attention_mask sum, not supervised positions
            return float(loss.item()), int(batch["attention_mask"].sum())
    finally:
        model.train(prev)


def _load_backbone(device, dtype_arg="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda":
        cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        dtype = torch.bfloat16 if (dtype_arg == "auto" and cc >= 80) else getattr(torch, dtype_arg, torch.bfloat16)
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-2B", dtype=dtype, low_cpu_mem_usage=True,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model, dtype


def score_text_nll(model, proj, vis, user, assistant, tok, device, max_len=4096):
    """Teacher-forced NLL of one (image, user, assistant) triple. No sampling."""
    from vision_adapter.core import make_collate

    coll = make_collate(tok, tok.pad_token_id, max_len=max_len, vision_dim=4096)
    batch = coll([{"vis": vis, "user": user, "assistant": assistant, "g": "eval"}])
    loss, _ = forward_loss(model, proj, batch, device)
    return loss


def discrimination_pairs(rows):
    """Non-overlapping consecutive pairs; leftover row dropped."""
    return [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1, 2)]


def _run_discrimination(model, tok, ds, args, llm_dim, dtype, device, data_dir, final_path):
    """Own-vs-swapped NLL on held-out rows (final ckpt only).

    acc_img (same text, own vs swapped image) isolates vision: text priors
    cancel. acc_txt (same image, own vs swapped text) is reported as sanity.
    Rows with n_vis>2000 skipped (VRAM cap, documented).
    """
    from vision_adapter.core import build_projector

    sd = torch.load(final_path, map_location=device, weights_only=False)
    proj = build_projector(4096, llm_dim, variant=args.variant_final).to(
        device, dtype=dtype if dtype != torch.float16 else torch.float32)
    proj.load_state_dict(sd.get("proj", sd))
    proj.eval()
    items, skipped = [], 0
    for it in ds:
        if int(it["vis"].shape[0]) > 2000:
            skipped += 1
            continue
        items.append(it)
        if len(items) >= 2 * args.n:
            break
    pairs = discrimination_pairs(items)
    print(f"[eval] {len(pairs)} pairs ({len(items)} rows, {skipped} monsters skipped)", flush=True)
    assert pairs, "no pairs collected — held-out shards unreachable?"
    img_wins = img_trials = txt_wins = txt_trials = 0
    m_img = m_txt = 0.0
    for a, b in pairs:
        s_aa = score_text_nll(model, proj, a["vis"], a["user"], a["assistant"], tok, device)
        s_ab = score_text_nll(model, proj, b["vis"], a["user"], a["assistant"], tok, device)
        s_bb = score_text_nll(model, proj, b["vis"], b["user"], b["assistant"], tok, device)
        s_ba = score_text_nll(model, proj, a["vis"], b["user"], b["assistant"], tok, device)
        img_wins += (s_aa < s_ab) + (s_bb < s_ba)
        img_trials += 2
        m_img += (s_ab - s_aa) + (s_ba - s_bb)
        txt_wins += (s_aa < s_ba) + (s_bb < s_ab)
        txt_trials += 2
        m_txt += (s_ba - s_aa) + (s_ab - s_bb)
    out = {"pairs": len(pairs), "skipped_monsters": skipped,
           "acc_img": img_wins / img_trials, "margin_img": m_img / img_trials,
           "acc_txt": txt_wins / txt_trials, "margin_txt": m_txt / txt_trials,
           "gate_img_60pct": bool(img_wins / img_trials >= 0.60)}
    (data_dir / "eval_discrimination.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


def _resolve_ckpt(local: str | None, repo: str, name: str) -> str:
    """Local ckpt path wins (diag ckpts never pushed); else HF download."""
    if local and Path(local).is_file():
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, name, repo_type="model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-repo", default="keypa/vision-adapter-probe-checkpoints")
    ap.add_argument("--ckpt-final", default="projector_final_4000.pt")
    ap.add_argument("--ckpt-base", default="projector_step400.pt")
    ap.add_argument("--local-final", default=None, help="local ckpt path (preferred over HF)")
    ap.add_argument("--local-base", default=None, help="local ckpt path (preferred over HF)")
    ap.add_argument("--variant-final", default=None, help="hourglass|scaled (default: env)")
    ap.add_argument("--variant-base", default=None, help="hourglass|scaled (default: env)")
    ap.add_argument("--mode", choices=("loss", "discrimination"), default="loss",
                    help="loss: mean NLL base vs final; discrimination: own-vs-swapped image/text NLL (final only)")
    ap.add_argument("--n", type=int, default=60,
                    help="loss: rows; discrimination: pairs (2 rows each, n_vis<=2000, monsters skipped)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    from vision_adapter.core import build_projector, make_collate
    from vision_adapter.data.stream import (
        EmbStreamDataset,
        build_epoch_plan,
        build_key_index,
        fetch_manifest,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "cache"

    print(f"[eval] resolving final {args.ckpt_final} (local preferred)", flush=True)
    final_path = _resolve_ckpt(args.local_final, args.ckpt_repo, args.ckpt_final)
    base_path = None
    if args.mode == "loss":
        print(f"[eval] resolving base {args.ckpt_base} (local preferred)", flush=True)
        base_path = _resolve_ckpt(args.local_base, args.ckpt_repo, args.ckpt_base)

    print("[eval] loading backbone", flush=True)
    tok, model, dtype = _load_backbone(device, args.dtype)
    llm_dim = int(getattr(model.config, "text_config", model.config).hidden_size)

    print(f"[eval] manifest + key index over {sorted(HELDOUT_SHARDS)}", flush=True)
    rows = fetch_manifest(cache_dir=str(cache_dir))
    order = sorted(HELDOUT_SHARDS)
    index = build_key_index(order, cache_dir=str(cache_dir))
    held = select_heldout_rows(rows, index, HELDOUT_SHARDS, n=args.n, seed=0)
    print(f"[eval] {len(held)} held-out rows (requested {args.n})", flush=True)
    assert held, "no held-out rows resolved — shards missing from index?"

    plan = build_epoch_plan(rows, index, sample_size=len(held), seed=0, excluded_shards=set())
    plan = {sf: v for sf, v in plan.items() if sf in HELDOUT_SHARDS}
    collate = make_collate(tok, tok.pad_token_id, max_len=4096, vision_dim=4096)
    ds = EmbStreamDataset(plan, order, rg_cache_dir=str(cache_dir / "rg_cache"))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, drop_last=False, collate_fn=collate, num_workers=0)

    if args.mode == "discrimination":
        _run_discrimination(model, tok, ds, args, llm_dim, dtype, device, data_dir,
                            final_path)
        return

    results = {}
    variants = {"base": args.variant_base, "final": args.variant_final}
    for name, path in (("base", base_path), ("final", final_path)):
        sd = torch.load(path, map_location=device, weights_only=False)
        proj = build_projector(4096, llm_dim, variant=variants[name]).to(
            device, dtype=dtype if dtype != torch.float16 else torch.float32)
        proj.load_state_dict(sd.get("proj", sd))
        proj.eval()
        losses, tokens = [], 0
        for batch in loader:
            loss, tok_n = forward_loss(model, proj, batch, device)
            losses.append(loss)
            tokens += tok_n
        results[name] = {"mean_loss": sum(losses) / len(losses), "batches": len(losses), "tokens": tokens}
        print(f"[eval] {name}: mean_loss={results[name]['mean_loss']:.4f} over {len(losses)} batches", flush=True)

    rel = (results["base"]["mean_loss"] - results["final"]["mean_loss"]) / results["base"]["mean_loss"]
    out = {"n_rows": len(held), "base": results["base"], "final": results["final"],
           "rel_improvement": rel, "gate_10pct": bool(rel >= 0.10)}
    (data_dir / "eval_heldout.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
