#!/usr/bin/env python3
"""Attention micro-bench apportionment (NEXT_STEPS §3.1 / Phase 5).

Synthetic inputs_embeds + block masks, shapes (B=16, L in 2k/5.4k/8k/9.6k)
x densities (1.0 uniform / 0.7 bucketed / 0.25 adversarial). 3 warmup + 10
timed steps (never step 0), per-layer-type forward-hook attribution,
peak allocated. ckpt point at (16, 8k, 0.25).

No Flex code here — measurement first, kill criteria decide (§3.4).

Usage (Molab GPU):
    TORCH_LOGS=sdpa python scripts/bench_attention.py --out data/bench_attention.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_shape(model, device, B, L, density=1.0, ckpt=False, steps=10, warmup=3):
    """One timed (fwd+bwd) point with per-layer-type attribution."""
    H = int(getattr(model.config, "text_config", model.config).hidden_size)
    tc = getattr(model.config, "text_config", model.config)
    n_layers = len(model.model.layers)
    layer_types = list(getattr(tc, "layer_types", ["unknown"] * n_layers))
    if len(layer_types) < n_layers:
        layer_types = (layer_types * n_layers)[:n_layers]
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    was_training = model.training
    model.train()
    if ckpt and device == "cuda":
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass
    k = max(1, int(L * density))
    try:
        times: dict[str, float] = {}
        handles = []

        def _mk_pre(store):
            def _pre(mod, inp):
                _sync(device)
                store[0] = time.perf_counter()
            return _pre

        def _mk_post(store, typ):
            def _post(mod, inp, out):
                _sync(device)
                times[typ] = times.get(typ, 0.0) + (time.perf_counter() - store[0])
            return _post

        for i, layer in enumerate(model.model.layers):
            store = [0.0]
            typ = layer_types[i] if i < len(layer_types) else "unknown"
            handles.append(layer.register_forward_pre_hook(_mk_pre(store)))
            handles.append(layer.register_forward_hook(_mk_post(store, typ)))
        fwd_ticks, bwd_ticks = [], []
        peak = 0.0
        for s in range(warmup + steps):
            g = torch.Generator().manual_seed(s)
            x = torch.randn(B, L, H, dtype=dtype, device=device, generator=g).requires_grad_(True)
            mask = torch.zeros(B, L, dtype=torch.long, device=device)
            mask[:, :k] = 1
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _sync(device)
            t0 = time.perf_counter()
            out = model.model(inputs_embeds=x, attention_mask=mask)
            _sync(device)
            t1 = time.perf_counter()
            out.last_hidden_state.sum().backward()
            _sync(device)
            t2 = time.perf_counter()
            if device == "cuda" and torch.cuda.is_available():
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
            if s >= warmup:
                fwd_ticks.append((t1 - t0) * 1000)
                bwd_ticks.append((t2 - t1) * 1000)
        return {"B": B, "L": L, "density": density, "ckpt": bool(ckpt),
                "fwd_ms": round(sum(fwd_ticks) / len(fwd_ticks), 2),
                "bwd_ms": round(sum(bwd_ticks) / len(bwd_ticks), 2),
                "by_type_ms": {t: round(v / steps, 2) for t, v in times.items()},
                "peak_alloc_mb": round(peak, 1)}
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)


SHAPES = [(16, 2000, 1.0), (16, 2000, 0.7), (16, 5400, 1.0), (16, 5400, 0.7),
          (16, 8000, 1.0), (16, 8000, 0.7), (16, 8000, 0.25), (16, 9600, 0.25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench_attention.json")
    ap.add_argument("--ckpt-point", action="store_true",
                    help="also run the (16, 8k, 0.25) ckpt-ON gate point")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bench] loading Qwen3.5-2B on {device} (frozen)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-2B", dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True, device_map=device if device == "cuda" else None)
    for p in model.parameters():
        p.requires_grad_(False)
    recs = []
    for B, L, d in SHAPES:
        try:
            r = bench_shape(model, device, B, L, density=d)
        except RuntimeError as e:
            r = {"B": B, "L": L, "density": d, "ckpt": False, "error": f"{type(e).__name__}: {e}"}
        recs.append(r)
        print(f"[bench] {r}", flush=True)
    if args.ckpt_point:
        r = bench_shape(model, device, 16, 8000, density=0.25, ckpt=True)
        r["gate"] = "ckpt-ON (16,8k,adv)"
        recs.append(r)
        print(f"[bench] {r}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(recs, indent=2))
    print(f"[bench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
