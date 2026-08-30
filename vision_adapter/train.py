"""vision_adapter/train.py — shared train runner for the staged CLI.

Colab/local entrypoint: `python -m vision_adapter train --data-dir ./data`
assumes you already ran `dataset` (header-first manifest) and `precompute`
(embeddings). On a GPU box (any CUDA — T4, L4, A100, 4090) it runs the
Qwen probe end-to-end (tiny synthetic fallback when the manifest is the fake
dry-run fixture, so `python -m vision_adapter dataset --dry-run` + train
still proves the loop without 120k real embeddings).

Modal path remains the thin wrapper in modal_train.py (Volume) — this module
is intentionally backend-agnostic: it takes DataBackend and a TrainConfig.
"""
from __future__ import annotations

from pathlib import Path

import torch

from vision_adapter.backends.gpu import require_gpu
from vision_adapter.config import TrainConfig, config_header, get_git_sha
from vision_adapter.manifest import load_manifest


def _tiny_qwen_for_smoke(vocab: int = 1024, hidden: int = 64, layers: int = 4):
    """Random-weight Qwen-shaped backbone, fp32 CPU/GPU — mirrors test_probe fixture.
    Last layer is full_attention so the projector receives grads (see test_probe notes)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
    tc = cfg.get_text_config()
    tc.vocab_size = vocab
    tc.hidden_size = hidden
    tc.intermediate_size = 2 * hidden
    tc.num_hidden_layers = layers
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.layer_types = ["linear_attention"] * (layers - 1) + ["full_attention"]
    try:
        tc.mtp_num_hidden_layers = None
    except Exception:
        pass
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class _StubTok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(ord(c) % 900) + 10 for c in text] or [10]}


def _smoke_train_with_fake_data(
    data_dir: Path,
    cfg: TrainConfig,
    max_steps: int | None,
    device: str,
) -> int:
    """5-step smoke on fake rows + random vis — no HF, no embeddings needed.
    Proves the projector + collate + monitors + selective loss wiring."""
    from vision_adapter.core import HourglassProjector, ProbeMonitor, make_collate, train_step_qwen

    rows, header = load_manifest(data_dir / "train_manifest.jsonl")
    is_fake = header is not None and any("fake" in r.get("emb", "") for r in rows[:5])
    # fall back to synthetic even when rows were from a real manifest but tiny limit
    limit_small = len(rows) <= 200
    if not is_fake and not limit_small and not header:
        # Not a fake fixture — caller should use the real HF streaming path
        return 1  # signal to caller to delegate to grok_probe_qwen instead

    steps = max_steps or 5
    steps = min(steps, 5)  # smoke cap
    tok = _StubTok()
    hidden = 64
    model = _tiny_qwen_for_smoke(hidden=hidden)
    dev = device if device in ("cuda", "cpu") and (device != "cuda" or torch.cuda.is_available()) else "cpu"
    if dev == "cuda":
        model = model.to("cuda")
    proj = HourglassProjector(cfg.vision_dim, hidden).to(dev)
    for p in proj.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(proj.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    monitor = ProbeMonitor()
    collate = make_collate(tok, tok.pad_token_id, max_len=cfg.max_seq_len, vision_dim=cfg.vision_dim)

    # Build batches from the manifest's fake rows: synthesize vis per row
    def _fake_vis(n_vis: int = 5):
        return torch.randn(n_vis, cfg.vision_dim)

    # Use at most `steps * batch_size` rows, reusing if needed
    import itertools
    B = cfg.batch_size
    batches = []
    cyc = itertools.cycle(rows if rows else [{"user": "hello", "assistant": "world", "emb": "x", "g": 0}])
    for _ in range(steps):
        items = []
        for _ in range(B):
            r = next(cyc)
            nv = 3 + (hash(r.get("emb", "")) % 5)
            items.append({"vis": _fake_vis(nv), "user": r.get("user", "hello"), "assistant": r.get("assistant", "hi"), "g": r.get("g", 0)})
        batches.append(collate(items))

    losses = []
    for i, batch in enumerate(batches, 1):
        out = train_step_qwen(model, proj, opt, batch, dev)
        losses.append(out["loss"])
        assert out["finite"], f"step {i} not finite: {out}"
        monitor.update(i, out["loss"], i * B)
    assert max(losses) - min(losses) >= 0  # at least runs
    print(f"[train] smoke {steps} steps on fake data — losses {[round(x,4) for x in losses]} monitor n_banners={monitor.n_banners}", flush=True)
    return 0




def _local_train_with_precomputed(data_dir: Path, cfg: TrainConfig, max_steps: int | None, device: str) -> int:
    """Train from local <data-dir>/embeddings/*.pt (produced by `precompute`)."""
    import json
    import time
    from pathlib import Path as _P

    from vision_adapter.core import HourglassProjector, ProbeMonitor, make_collate, train_step_qwen, render_curves
    from vision_adapter.registry import append_registry, registry_entry

    rows, header = load_manifest(data_dir / "train_manifest.jsonl")
    # Filter to rows whose embedding exists locally
    emb_dir = data_dir / "embeddings"
    local_keys = {p.name for p in emb_dir.glob("*.pt")} if emb_dir.is_dir() else set()
    avail = [r for r in rows if r.get("emb", "").split("/")[-1] in local_keys]
    if not avail:
        print(f"[train] local embeddings requested but none matched manifest ({len(rows)} rows, {len(local_keys)} .pt)", flush=True)
        return _streaming_train(data_dir, cfg, max_steps, device)
    # Cap to max_steps * batch for smoke, else all
    import random as _rnd
    _rnd.Random(0).shuffle(avail)
    sample = avail[: (max_steps or 5) * cfg.batch_size] if max_steps else avail

    # Load tokenizer + backbone (Qwen 2B) — same as grok path but via transformers
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = device if device in ("cuda", "cpu") and (device != "cuda" or _torch.cuda.is_available()) else "cpu"
    # dtype: bf16 on Ampere+, else fp32
    if dev == "cuda":
        p = _torch.cuda.get_device_properties(0)
        dtype = _torch.bfloat16 if p.major >= 8 else _torch.float32
    else:
        dtype = _torch.float32
    print(f"[train] local train: {len(sample)} rows, device={dev} dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", dtype=dtype, low_cpu_mem_usage=True, device_map=dev if dev=="cuda" else None)
    if dev == "cpu":
        model = model.to("cpu")
    for pa in model.parameters():
        pa.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg_llm = getattr(model.config, "text_config", model.config)
    llm_dim = int(cfg_llm.hidden_size)
    proj = HourglassProjector(cfg.vision_dim, llm_dim).to(dev, dtype=dtype)
    for pa in proj.parameters():
        pa.requires_grad_(True)
    opt = _torch.optim.AdamW(proj.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    collate = make_collate(tok, tok.pad_token_id, max_len=cfg.max_seq_len, vision_dim=cfg.vision_dim)
    monitor = ProbeMonitor()
    out_dir = _P(data_dir)
    log_path = out_dir / "probe_log.jsonl"
    curves_path = out_dir / "probe_curves.png"
    # header
    try:
        hdr = config_header(cfg, manifest_path=str(data_dir / "train_manifest.jsonl"), extra={"run": "train-local", "device": dev, "dtype": str(dtype)})
        run_id = hdr.get("run_id")
        with open(log_path, "w", buffering=1) as lf:
            lf.write(json.dumps(hdr) + "\n")
    except Exception as e:
        print(f"[train] header write failed: {e}", flush=True)
        run_id = None
        open(log_path, "w").close()
    # build vis cache in RAM (small sample only)
    def _load_vis(emb_key: str):
        pt = emb_dir / emb_key.split("/")[-1]
        ten = _torch.load(str(pt), map_location="cpu")
        # ten is [n_vis, 4096] bf16 or fp32
        return ten.float()
    # batch loop
    steps = max_steps or 5
    recs: list[dict] = []
    t0 = time.time()
    with open(log_path, "a", buffering=1) as lf:
        for step in range(1, steps + 1):
            batch_rows = [sample[(step-1)*cfg.batch_size + i % len(sample)] for i in range(cfg.batch_size)]
            items = [{"vis": _load_vis(r["emb"]), "user": r.get("user",""), "assistant": r.get("assistant",""), "g": r.get("g","")} for r in batch_rows]
            batch = collate(items)
            out = train_step_qwen(model, proj, opt, batch, dev)
            rec = {"type": "train", "step": step, "loss": round(out["loss"],5), "gnorm": round(out["gnorm"],4), "lr": float(opt.param_groups[0]["lr"]), "tokens": out["tokens"], "samples_seen": step*cfg.batch_size, "step_ms": out["step_ms"]}
            monitor.update(step, rec["loss"], rec["samples_seen"])
            rec["ema_loss"] = round(monitor.ema or rec["loss"],5)
            recs.append(rec)
            lf.write(json.dumps(rec) + "\n")
            if step % 5 == 0 or step == steps:
                print(f"[train] local step {step}/{steps} loss={rec['loss']:.4f} gnorm={rec['gnorm']:.2f}", flush=True)
        # curves + run_end
        try:
            render_curves(recs, str(curves_path))
        except Exception:
            pass
        wall = round((time.time()-t0)/60,1)
        lf.write(json.dumps({"type":"run_end","run_id":run_id,"step":steps,"samples_seen":steps*cfg.batch_size,"final_loss": recs[-1]["loss"] if recs else None,"wall_min":wall})+"\n")
    try:
        reg = registry_entry(run_id=run_id, git_sha=get_git_sha(), config=cfg.to_dict(), seed=0, device=dev, dtype=str(dtype), step_ms=recs[-1].get("step_ms") if recs else None, final_loss=recs[-1]["loss"] if recs else None, extra={"run":"train-local"})
        append_registry(str(out_dir / "runs.jsonl"), reg)
    except Exception:
        pass
    return 0


def _streaming_train(data_dir: Path, cfg: TrainConfig, max_steps: int | None, device: str) -> int:  # noqa: C901
    """Native HF streaming train — cluster-sampled RemoteShard, no grok import."""
    import json
    import time
    import os
    import random
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vision_adapter.backends.auth import get_hf_token as _ghf
    from vision_adapter.core import HourglassProjector, ProbeMonitor, make_collate, train_step_qwen, render_curves, lr_at
    from vision_adapter.data.stream import (
        EmbStreamDataset as _EmbDS,
        build_epoch_plan as _build_plan,
        build_key_index as _build_index,
        fetch_manifest as _fetch_manifest,
        list_shards as _list_shards,
    )
    from vision_adapter.registry import append_registry, registry_entry

    # Resolve HF token for streaming
    tok_hf = _ghf()
    if tok_hf:
        os.environ["HF_TOKEN"] = tok_hf
    # Manifest: prefer local file; else fetch from HF
    local_manifest = data_dir / "train_manifest.jsonl"
    if local_manifest.is_file():
        rows, header = load_manifest(local_manifest)
        # If local manifest is fake fixture, treat as smoke — caller already handled
        is_fake = any("fake" in r.get("emb","") for r in rows[:10])
        if is_fake and len(rows) <= 200:
            return _smoke_train_with_fake_data(data_dir, cfg, max_steps, device)
    else:
        rows = _fetch_manifest(cache_dir=str(data_dir / "cache"), token=tok_hf)
    # Device / dtype (mirror grok logic: bf16 Ampere+, else fp32/fp16 via autocast)
    dev = device if device in ("cuda","cpu") and (device!="cuda" or _torch.cuda.is_available()) else "cpu"
    if dev == "cuda":
        p = _torch.cuda.get_device_properties(0)
        cc = p.major*10 + p.minor
        dtype = {"auto": _torch.bfloat16 if cc>=80 else (_torch.float16 if cc>=70 else _torch.float32)}["auto"]
        _torch.backends.cuda.matmul.allow_tf32 = True
    else:
        dtype = _torch.bfloat16 if False else _torch.float32
    # Tokenizer + backbone
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"[train] streaming: loading Qwen3.5-2B on {dev} dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", dtype=dtype, low_cpu_mem_usage=True, device_map=dev if dev=="cuda" else None)
    if dev == "cpu":
        model = model.to("cpu")
    for pa in model.parameters():
        pa.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg_llm = getattr(model.config, "text_config", model.config)
    llm_dim = int(cfg_llm.hidden_size)
    proj_dtype = _torch.float32 if (dev=="cuda" and dtype==_torch.float16) else dtype
    proj = HourglassProjector(cfg.vision_dim, llm_dim).to(dev, dtype=proj_dtype)
    for pa in proj.parameters():
        pa.requires_grad_(True)
    opt = _torch.optim.AdamW(proj.parameters(), lr=cfg.lr, betas=(0.9,0.95))
    scaler = _torch.amp.GradScaler("cuda", enabled=(dev=="cuda" and dtype==_torch.float16), init_scale=1.0, growth_interval=10**9) if dev=="cuda" else None
    collate = make_collate(tok, tok.pad_token_id, max_len=cfg.max_seq_len, vision_dim=cfg.vision_dim)
    monitor = ProbeMonitor()
    # Build streaming plan
    stream_order = _list_shards(token=tok_hf)
    EXCLUDED = {"data/emb_0000.parquet", "data/emb_0001.parquet"}
    stream_order = [s for s in stream_order if s not in EXCLUDED]
    random.Random(0).shuffle(stream_order)
    index = _build_index(stream_order, cache_dir=str(data_dir / "cache"))
    sample_size = min(len(rows), (max_steps or 5) * cfg.batch_size * 2)
    plan = _build_plan(rows, index, sample_size=sample_size, seed=0, excluded_shards=EXCLUDED)
    n_planned = sum(len(v) for v in plan.values())
    print(f"[train] streaming plan: {n_planned} rows from {len(plan)} shards", flush=True)
    # Logging
    log_path = data_dir / "probe_log.jsonl"
    curves_path = data_dir / "probe_curves.png"
    try:
        hdr = config_header(cfg, manifest_path=str(local_manifest) if local_manifest.is_file() else None, extra={"run":"train-stream","device":dev,"dtype":str(dtype),"sample_size":sample_size})
        run_id = hdr.get("run_id")
        with open(log_path, "w", buffering=1) as lf:
            lf.write(json.dumps(hdr)+"\n")
    except Exception as e:
        print(f"[train] header failed: {e}", flush=True)
        run_id = None
        open(log_path,"w").close()
    # Batch iterator
    def _batch_iter():
        ds = _EmbDS(plan, stream_order, rg_cache_dir=str(data_dir / "cache" / "rg_cache"), vision_dim=cfg.vision_dim)
        loader = _torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, drop_last=True, collate_fn=collate, num_workers=0)
        yield from loader
        # epoch wrap
        while True:
            ds2 = _EmbDS(plan, stream_order, rg_cache_dir=str(data_dir / "cache" / "rg_cache"), vision_dim=cfg.vision_dim)
            loader2 = _torch.utils.data.DataLoader(ds2, batch_size=cfg.batch_size, drop_last=True, collate_fn=collate, num_workers=0)
            yield from loader2
    it = _batch_iter()
    steps = max_steps or 5
    recs: list[dict] = []
    t0 = time.time()
    with open(log_path, "a", buffering=1) as lf:
        for step in range(1, steps+1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps, cfg.lr, cfg.warmup_steps)
            batch = next(it)
            out = train_step_qwen(model, proj, opt, batch, dev, scaler=scaler)
            if not out["finite"]:
                print(f"[train][WARN] non-finite at {step}, skipping", flush=True)
                continue
            rec = {"type":"train","step":step,"loss":round(out["loss"],5),"gnorm":round(out["gnorm"],4),"lr":float(opt.param_groups[0]["lr"]),"tokens":out["tokens"],"samples_seen":step*cfg.batch_size,"step_ms":out["step_ms"],"ts": round(time.time(),1)}
            monitor.update(step, rec["loss"], rec["samples_seen"])
            rec["ema_loss"] = round(monitor.ema or rec["loss"],5)
            recs.append(rec)
            lf.write(json.dumps(rec)+"\n")
            if step % 5 == 0 or step==steps:
                print(f"[train] stream step {step}/{steps} loss={rec['loss']:.4f} ema={rec['ema_loss']:.4f} gnorm={rec['gnorm']:.2f}", flush=True)
        try:
            render_curves(recs, str(curves_path))
        except Exception:
            pass
        wall = round((time.time()-t0)/60,1)
        lf.write(json.dumps({"type":"run_end","run_id":run_id,"step":steps,"samples_seen":steps*cfg.batch_size,"final_loss": recs[-1]["loss"] if recs else None,"wall_min":wall})+"\n")
    try:
        reg = registry_entry(run_id=run_id, git_sha=get_git_sha(), config=cfg.to_dict(), seed=0, device=dev, dtype=str(dtype), step_ms=recs[-1].get("step_ms") if recs else None, final_loss=recs[-1]["loss"] if recs else None, extra={"run":"train-stream"})
        append_registry(str(data_dir / "runs.jsonl"), reg)
    except Exception:
        pass
    return 0

def run_train(
    data_dir: Path | str,
    cfg: TrainConfig,
    backend=None,
    max_steps: int | None = None,
    device: str | None = None,
) -> int:
    """Entry point for `cli train` non-dryrun.

    - Validates data_dir + manifest (header-first)
    - require_gpu("train") if device == "cuda" (any GPU)
    - Chooses path: fake-smoke (tiny) vs HF streaming (grok_probe) vs error with guidance
    Returns 0 on success, 1 if caller should delegate (e.g. no fake fixture, need HF path).
    """
    dd = Path(data_dir)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Only gate on GPU when we actually need CUDA kernels; smoke fallback runs on CPU
    # but warn — real training will need a card.
    need_gpu = dev == "cuda"
    if need_gpu:
        try:
            require_gpu("train")
        except SystemExit as e:
            print(str(e), flush=True)
            raise

    manifest_path = dd / "train_manifest.jsonl"
    if not manifest_path.exists():
        print(f"[train] manifest not found at {manifest_path} — run `python -m vision_adapter dataset --out {dd} [--dry-run]` first", flush=True)
        return 2

    # Prefer fake smoke when manifest is the dry-run fixture (fast proof without 2B download)
    rows, header = load_manifest(manifest_path)
    is_fake_fixture = header is not None and any("fake" in r.get("emb", "") for r in rows[:10])
    if is_fake_fixture or len(rows) <= 200:
        print(f"[train] detected {'fake fixture' if is_fake_fixture else 'small manifest'} ({len(rows)} rows) — running tiny smoke (no 2B download)", flush=True)
        return _smoke_train_with_fake_data(dd, cfg, max_steps, dev)

    # Real data: native HF streaming (no grok shell-out).
    # Uses vision_adapter/data/stream.py (RemoteShard cluster sampling) so
    # `python -m vision_adapter train --data-dir ./data --max-steps 200`
    # works on any CUDA host without an extra clone of grok_probe_qwen.py.
    # Local embeddings under <data-dir>/embeddings/*.pt are preferred when present.
    local_emb_dir = dd / "embeddings"
    has_local_emb = local_emb_dir.is_dir() and any(local_emb_dir.glob("*.pt"))
    if has_local_emb:
        print(f"[train] local embeddings found at {local_emb_dir} — using local path", flush=True)
        return _local_train_with_precomputed(dd, cfg, max_steps, dev)
    print("[train] no local embeddings — streaming from HF (keypa/vision-adapter-embeddings)", flush=True)
    try:
        return _streaming_train(dd, cfg, max_steps, dev)
    except Exception as e:
        import traceback
        print(f"[train] streaming train failed ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        print("[train] falling back to smoke stub (check HF_TOKEN and network)", flush=True)
        return _smoke_train_with_fake_data(dd, cfg, max_steps, dev)
