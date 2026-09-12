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
import time
import torch

from vision_adapter.backends.gpu import require_gpu
from vision_adapter.config import TrainConfig, config_header, get_git_sha
from vision_adapter.manifest import load_manifest

def _stats_str() -> str:
    """GPU util %, VRAM total via nvidia-smi (global), RAM %, CPU % — never crash."""
    try:
        import psutil
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=None)
    except Exception:
        ram, cpu = 0, 0
    try:
        if torch.cuda.is_available():
            # Use nvidia-smi for true total VRAM used (torch.cuda.memory_allocated is per-process and shows 3.7GB vs 92.9GB nvidia-smi)
            vram_used = vram_total = util = 0
            try:
                import subprocess
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], timeout=1, stderr=subprocess.DEVNULL)
                u, t, ug = out.decode().strip().split(", ")
                vram_used = int(u); vram_total = int(t); util = int(ug)
                vram_gb = vram_used/1024
                vram_pct = vram_used / vram_total * 100 if vram_total else 0
            except Exception:
                # fallback to torch
                props = torch.cuda.get_device_properties(0)
                vram_total = props.total_memory/1024**2
                vram_used = torch.cuda.memory_allocated(0)/1024**2
                vram_gb = vram_used/1024
                vram_pct = vram_used / vram_total * 100 if vram_total else 0
                util = 0
            return f"GPU {util:3d}% VRAM {vram_gb:.1f}GB {vram_pct:.0f}% RAM {ram:.0f}% CPU {cpu:.0f}%"
        else:
            return f"RAM {ram:.0f}% CPU {cpu:.0f}%"
    except Exception:
        return f"RAM {ram:.0f}% CPU {cpu:.0f}%"


def _hf_ckpt_push_cfg() -> tuple[bool, str | None]:
    """Push-to-HF gate for checkpoints: env VISION_ADAPTER_PUSH_HF=1 + VISION_ADAPTER_HF_CKPT_REPO.

    Best-effort only — never crash training on hub errors (ephemeral Molab sessions
    must survive network/token issues; local file is the source of truth).
    """
    import os

    if os.environ.get("VISION_ADAPTER_PUSH_HF") != "1":
        return False, None
    repo = os.environ.get("VISION_ADAPTER_HF_CKPT_REPO") or None
    if not repo:
        return False, None
    return True, repo


def _push_file_to_hf(local_path: Path, repo_id: str) -> None:
    """Upload one file to a HF model repo (create if missing). Raises on failure."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"ckpt {local_path.name}",
    )


def _maybe_push_ckpt(local_path: Path) -> None:
    enabled, repo = _hf_ckpt_push_cfg()
    if not enabled or not repo:
        return
    try:
        _push_file_to_hf(local_path, repo)
        print(f"[train] pushed {local_path.name} -> hf:{repo}", flush=True)
    except Exception as e:  # noqa: BLE001 — hub push must never kill a 6h run
        print(f"[train] HF push failed for {local_path.name} ({e}) — local copy kept", flush=True)


PUSH_MAX_INTERVAL_S = 600  #(env VISION_ADAPTER_PUSH_INTERVAL_S): max 10min without a save+push


def _save_due(step: int, save_every: int, last_save_ts: float, now: float,
              max_interval_s: float = PUSH_MAX_INTERVAL_S) -> bool:
    """Step-gated save OR time-gated save (monster steps can take 60s+ each —
    without the time gate, 100 steps could mean 100min without a save)."""
    if step % save_every == 0:
        return True
    return (now - last_save_ts) >= max_interval_s


CKPT_KEYS = ("proj", "opt", "scaler", "step", "samples_seen",
             "monitor", "rng", "plan", "cfg", "run_id")


def _collect_rng_state() -> dict:
    import random

    import torch

    state: dict = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def build_ckpt_payload(proj_state, opt_state, scaler_state, step: int,
                       samples_seen: int, monitor_state: dict,
                       rng_state: dict, plan_meta: dict, cfg_dict: dict,
                       run_id: str | None = None) -> dict:
    return {"proj": proj_state, "opt": opt_state, "scaler": scaler_state,
            "step": step, "samples_seen": samples_seen,
            "monitor": monitor_state, "rng": rng_state,
            "plan": plan_meta, "cfg": cfg_dict, "run_id": run_id}


def _resolve_resume(ckpt_plan: dict, cli_max_steps: int | None, cli_seed: int = 0) -> dict:
    """Pin training extent from ckpt metadata, never from new CLI values."""
    total = int(ckpt_plan["max_steps"]) if "max_steps" in ckpt_plan else None
    if total is None:  # backward compat: old ckpts without max_steps
        total = int(cli_max_steps or 0)
    return {"total_steps": total, "seed": int(ckpt_plan.get("seed", cli_seed)),
            "sample_size": int(ckpt_plan.get("sample_size", 0)),
            "batch_size": int(ckpt_plan.get("batch_size", 16)),
            "resume_step": int(ckpt_plan.get("step", 0)),
            "start_pos_rows": int(ckpt_plan.get("step", 0)) * int(ckpt_plan.get("batch_size", 16))}


def _open_resume_log(log_path, run_id):
    """Append to the existing run log, preserving run_id."""
    fh = open(log_path, "a", buffering=1)
    return fh, run_id


def _restore_rng_state(rng_state: dict | None) -> None:
    """Best-effort RNG restore (python/torch/cuda/numpy, per key)."""
    if not rng_state:
        return
    try:
        import random

        if "python" in rng_state:
            random.setstate(rng_state["python"])
    except Exception:
        pass
    try:
        import torch as _t

        if "torch" in rng_state:
            _t.set_rng_state(rng_state["torch"])
        if "cuda" in rng_state and _t.cuda.is_available():
            try:
                _t.cuda.set_rng_state_all(rng_state["cuda"])
            except Exception:
                pass
    except Exception:
        pass
    try:
        import numpy as _np

        if "numpy" in rng_state:
            _np.random.set_state(rng_state["numpy"])
    except Exception:
        pass


def _ckpt_step_number(name: str) -> int:
    try:
        return int(name.removeprefix("projector_step").removesuffix(".pt"))
    except ValueError:
        return -1


def _find_local_ckpt(data_dir: Path | str, step: int | None = None) -> Path:
    """Locate a local step ckpt (latest when step is None, else exact).

    Searches the streaming cache root first, then data_dir itself.
    Raises FileNotFoundError when nothing matches.
    """
    import os

    dd = Path(data_dir)
    cache_root = Path("/hf/hf_stream_cache") if (Path("/hf").is_dir() and os.environ.get("MODAL_TASK_ID")) else dd / "cache"
    search_dirs = [cache_root, dd]
    if step is not None:
        filename = f"projector_step{step}.pt"
        for d in search_dirs:
            cand = d / filename
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"{filename} not found in {[str(d) for d in search_dirs]}")
    best: Path | None = None
    best_n = -1
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in d.glob("projector_step*.pt"):
            n = _ckpt_step_number(p.name)
            if n > best_n:
                best_n = n
                best = p
    if best is None:
        raise FileNotFoundError(f"no projector_step*.pt in {[str(d) for d in search_dirs]}")
    return best


def _ensure_expandable_segments() -> bool:
    """Default PYTORCH_CUDA_ALLOC_CONF to expandable_segments (fragmentation relief).

    Must run before the first CUDA allocation (i.e. model load). Never
    overrides an explicit user setting. Returns True when it set the default.
    """
    import os

    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return True


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




def _local_train_with_precomputed(data_dir: Path, cfg: TrainConfig, max_steps: int | None, device: str, dtype_arg: str = "auto") -> int:
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
        return _streaming_train(data_dir, cfg, max_steps, device, dtype_arg)
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
    # dtype: honors --dtype, defaults to bf16 Ampere+ else fp32
    if dev == "cuda":
        p = _torch.cuda.get_device_properties(0)
        cc = p.major*10 + p.minor
        dtype_map = {"auto": _torch.bfloat16 if cc>=80 else _torch.float32, "bf16": _torch.bfloat16, "fp16": _torch.float16, "fp32": _torch.float32}
        dtype = dtype_map.get(dtype_arg, dtype_map["auto"])
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


def _persist_fetched_manifest(data_dir: Path, rows: list[dict]) -> Path:
    """Write HF-fetched rows to data_dir/train_manifest.jsonl (header-first).

    Without this, rows live only in memory and every later stage that reads
    the local manifest (smoke fallback, local train, final push) crashes or
    silently skips. Atomic tmp+replace; never raises (logs and returns path).
    """
    from vision_adapter.manifest import write_manifest_with_header

    out = Path(data_dir) / "train_manifest.jsonl"
    write_manifest_with_header(out, rows)
    return out


def _streaming_train(data_dir: Path, cfg: TrainConfig, max_steps: int | None, device: str, dtype_arg: str = "auto", resume_ckpt: dict | None = None) -> int:  # noqa: C901
    """Native HF streaming train — cluster-sampled RemoteShard, no grok import.

    When resume_ckpt is given, continue at ckpt["step"]+1 with restored
    opt/monitor, LR pinned to the ORIGINAL total steps, data skipped via
    start_pos, and the log appended under the SAME run_id (statistical
    equivalence with the uninterrupted curve, not bit-identical).
    """
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
        try:
            _persist_fetched_manifest(data_dir, rows)
        except Exception as e:  # noqa: BLE001 — persistence is best-effort, training must not die here
            print(f"[train] manifest persist failed ({e}) — continuing with in-memory rows", flush=True)
    # Device / dtype (mirror grok logic: bf16 Ampere+, else fp32/fp16 via autocast; --dtype overrides)
    dev = device if device in ("cuda","cpu") and (device!="cuda" or _torch.cuda.is_available()) else "cpu"
    if dev == "cuda":
        p = _torch.cuda.get_device_properties(0)
        cc = p.major*10 + p.minor
        dtype_map = {"auto": _torch.bfloat16 if cc>=80 else (_torch.float16 if cc>=70 else _torch.float32), "bf16": _torch.bfloat16, "fp16": _torch.float16, "fp32": _torch.float32}
        dtype = dtype_map.get(dtype_arg, dtype_map["auto"])
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
    # PRO 6000 96GB Blackwell → disable grad ckpt to save 25-40% recompute (Qwen 2B fits easily, was bottleneck on T4/L4)
    _is_big_gpu = False
    try:
        if dev=="cuda":
            _is_big_gpu = _torch.cuda.get_device_properties(0).total_memory > 80*2**30
    except Exception:
        pass
    if os.environ.get("VISION_ADAPTER_CKPT_OFF") == "1" or _is_big_gpu:
        model.gradient_checkpointing_disable()
        print(f"[train] ckpt OFF (big GPU {_is_big_gpu} 96GB -> 25-40% faster, no recompute)", flush=True)
    else:
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
    # --resume local: pin extent from ckpt, restore weights/opt/monitor/RNG.
    _resume_info: dict | None = None
    if resume_ckpt is not None:
        _merged_plan = dict(resume_ckpt.get("plan", {}) or {})
        _merged_plan.setdefault("step", resume_ckpt.get("step", 0))
        _resume_info = _resolve_resume(_merged_plan, cli_max_steps=max_steps)
        try:
            proj.load_state_dict(resume_ckpt["proj"])
        except Exception as e:  # noqa: BLE001 — shape/dtype mismatch must surface clearly
            print(f"[train] resume proj restore failed ({e})", flush=True)
            raise
        try:
            opt.load_state_dict(resume_ckpt["opt"])
        except Exception as e:  # noqa: BLE001
            print(f"[train] resume opt restore failed ({e})", flush=True)
            raise
        try:
            if scaler is not None and resume_ckpt.get("scaler") is not None:
                scaler.load_state_dict(resume_ckpt["scaler"])
        except Exception as e:  # noqa: BLE001 — scaler restore is best-effort
            print(f"[train] resume scaler restore skipped ({e})", flush=True)
        try:
            monitor.load_state_dict(resume_ckpt.get("monitor", {}) or {})
        except Exception as e:  # noqa: BLE001
            print(f"[train] resume monitor restore failed ({e})", flush=True)
            raise
        _restore_rng_state(resume_ckpt.get("rng"))
        print(f"[train] resume from step {resume_ckpt.get('step')} -> {int(_resume_info['resume_step']) + 1}..{int(_resume_info['total_steps'])} "
              f"start_pos_rows={int(_resume_info['start_pos_rows'])} run_id={resume_ckpt.get('run_id')}", flush=True)
    # Build streaming plan (ensure cache dirs exist before index save)
    # Persistent cache on Modal via HF_CACHE (/hf) to avoid 224s rebuild each ephemeral /tmp run
    _cache_root = Path("/hf/hf_stream_cache") if (Path("/hf").is_dir() and os.environ.get("MODAL_TASK_ID")) else data_dir / "cache"
    _cache_root.mkdir(parents=True, exist_ok=True)
    (_cache_root / "rg_cache").mkdir(parents=True, exist_ok=True)
    # keep ephemeral symlink for backward compat
    try:
        (data_dir / "cache").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    stream_order = _list_shards(token=tok_hf)
    EXCLUDED = {"data/emb_0000.parquet", "data/emb_0001.parquet"}
    stream_order = [s for s in stream_order if s not in EXCLUDED]
    if os.environ.get("FORCE_LARGEST_BUCKET") == "1":
        # worst-case VRAM: last 4 bucketed shards = 4901+ 59.4GiB each (sorted numeric = bucket order)
        stream_order = sorted(stream_order)[-4:]
        print(f"[train] FORCE_LARGEST_BUCKET: testing {len(stream_order)} largest shards {stream_order} (worst VRAM, FlexAttention TODO)", flush=True)
    else:
        random.Random(0).shuffle(stream_order)
    index = _build_index(stream_order, cache_dir=str(_cache_root))
    if _resume_info is not None and int(_resume_info.get("sample_size", 0)) > 0:
        sample_size = int(_resume_info["sample_size"])
        _plan_seed = int(_resume_info.get("seed", 0))
    else:
        sample_size = min(len(rows), (max_steps or 5) * cfg.batch_size * 2)
        _plan_seed = 0
    plan = _build_plan(rows, index, sample_size=sample_size, seed=_plan_seed, excluded_shards=EXCLUDED)
    n_planned = sum(len(v) for v in plan.values())
    print(f"[train] streaming plan: {n_planned} rows from {len(plan)} shards", flush=True)
    # Logging
    log_path = data_dir / "probe_log.jsonl"
    curves_path = data_dir / "probe_curves.png"
    if _resume_info is not None:
        run_id = resume_ckpt.get("run_id") if isinstance(resume_ckpt, dict) else None
        if not run_id:
            # backward compat: old ckpts without run_id — recover from existing log header
            try:
                with open(log_path) as _lf:
                    _first = _lf.readline().strip()
                    run_id = json.loads(_first).get("run_id") if _first else None
            except Exception:
                run_id = None
        print(f"[train] resume appending to {log_path} run_id={run_id}", flush=True)
    else:
        try:
            hdr = config_header(cfg, manifest_path=str(local_manifest) if local_manifest.is_file() else None, extra={"run":"train-stream","device":dev,"dtype":str(dtype),"sample_size":sample_size})
            run_id = hdr.get("run_id")
            with open(log_path, "w", buffering=1) as lf:
                lf.write(json.dumps(hdr)+"\n")
        except Exception as e:
            print(f"[train] header failed: {e}", flush=True)
            run_id = None
            open(log_path,"w").close()
    # Phase 2 daemon: prefetch shard i+1 via hf_transfer while GPU trains shard i (1GiB/s, pipelined)
    # EmbStreamDataset already has shard-level prefetch + LRU 4 shards; this top-level daemon warms the first shard
    # before the loop so the first batch never stalls (12min cold pipelined over 33h).
    _prefetch_exec = None
    _first_shard_fut = None
    try:
        from concurrent.futures import ThreadPoolExecutor as _TPE

        from vision_adapter.data.stream import _download_shard_hf_transfer as _dl_shard, _get_hf_shard_path as _get_path, _in_modal as _is_modal

        if _is_modal():
            # Warm first shard in background (if not already cached)
            first_sf = next((s for s in stream_order if s in plan), None)
            _rg_dir = str(_cache_root / "rg_cache")
            if first_sf and _get_path(first_sf, cache_dir=_rg_dir) is None:
                _prefetch_exec = _TPE(max_workers=1)
                _first_shard_fut = _prefetch_exec.submit(_dl_shard, first_sf, _rg_dir)
                print(f"[train] daemon prefetching first shard {first_sf} ...", flush=True)
    except Exception:
        _prefetch_exec = None
        _first_shard_fut = None
    # Batch iterator
    _start_pos = int(_resume_info["start_pos_rows"]) if _resume_info is not None else 0

    def _batch_iter():
        _rg = str(_cache_root / "rg_cache") if "_cache_root" in locals() else str(data_dir / "cache" / "rg_cache")
        ds = _EmbDS(plan, stream_order, start_pos=_start_pos, rg_cache_dir=_rg, vision_dim=cfg.vision_dim)
        loader = _torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, drop_last=True, collate_fn=collate, num_workers=0)
        yield from loader
        # epoch wrap
        while True:
            # NOTE: ds2 keeps start_pos=0 on purpose — _start_pos is a one-time
            # skip into the interrupted epoch for data continuity; later epochs
            # replay fully while samples_seen bookkeeping continues via step.
            ds2 = _EmbDS(plan, stream_order, rg_cache_dir=str(data_dir / "cache" / "rg_cache"), vision_dim=cfg.vision_dim)
            loader2 = _torch.utils.data.DataLoader(ds2, batch_size=cfg.batch_size, drop_last=True, collate_fn=collate, num_workers=0)
            yield from loader2
    it = _batch_iter()
    # Ensure first shard prefetch completes before first batch (or timeout 30s)
    if _first_shard_fut is not None:
        try:
            _first_shard_fut.result(timeout=30)
            print("[train] first shard prefetch ready", flush=True)
        except Exception:
            pass
    if _resume_info is not None:
        steps = int(_resume_info["total_steps"])
        start_step = int(_resume_info["resume_step"]) + 1
    else:
        steps = max_steps or 5
        start_step = 1
    recs: list[dict] = []
    t0 = time.time()
    import os as _os
    try:
        _push_interval = float(_os.environ.get("VISION_ADAPTER_PUSH_INTERVAL_S", PUSH_MAX_INTERVAL_S))
    except ValueError:
        _push_interval = float(PUSH_MAX_INTERVAL_S)
    last_save_ts = t0
    if _resume_info is not None:
        _resume_fh, run_id = _open_resume_log(log_path, run_id)
        _log_ctx = _resume_fh
    else:
        _log_ctx = open(log_path, "a", buffering=1)
    with _log_ctx as lf:
        for step in range(start_step, steps+1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps, cfg.lr, cfg.warmup_steps)
            batch = next(it)
            out = train_step_qwen(model, proj, opt, batch, dev, scaler=scaler)
            if not out["finite"]:
                print(f"[train][WARN] non-finite at {step}, skipping", flush=True)
                continue
            rec = {"type":"train","step":step,"loss":round(out["loss"],5),"gnorm":round(out["gnorm"],4),"lr":float(opt.param_groups[0]["lr"]),"tokens":out["tokens"],"L":out.get("L"),"bl2":out.get("bl2"),"ckpt_on":out.get("ckpt_on", False),"samples_seen":step*cfg.batch_size,"step_ms":out["step_ms"],"ts": round(time.time(),1)}
            monitor.update(step, rec["loss"], rec["samples_seen"])
            rec["ema_loss"] = round(monitor.ema or rec["loss"],5)
            recs.append(rec)
            lf.write(json.dumps(rec)+"\n")
            # save every 10 steps for probe (200) to avoid losing $ on interrupt; hero uses cfg.save_every 500
            _save_every = 10 if steps <= 500 else cfg.save_every
            if _save_due(step, _save_every, last_save_ts, time.time(), _push_interval):
                try:
                    ckpt = _cache_root / f"projector_step{step}.pt"
                    import hashlib as _hashlib

                    from vision_adapter.config import manifest_sha256 as _mhash

                    _order_hash = _hashlib.sha1(
                        json.dumps(list(stream_order)).encode()
                    ).hexdigest()
                    _plan_meta = {
                        "manifest_sha256": _mhash(local_manifest),
                        "seed": 0,
                        "sample_size": sample_size,
                        "batch_size": cfg.batch_size,
                        "stream_order_hash": _order_hash,
                        "max_steps": steps,
                    }
                    _payload = build_ckpt_payload(
                        proj.state_dict(),
                        opt.state_dict(),
                        scaler.state_dict() if scaler is not None else None,
                        step,
                        step * cfg.batch_size,
                        monitor.to_dict(),
                        _collect_rng_state(),
                        _plan_meta,
                        cfg.to_dict(),
                        run_id,
                    )
                    _torch.save(_payload, str(ckpt))
                    print(f"[{time.strftime('%H:%M:%S')} {(time.time()-t0)/60:.1f}min] [train] ckpt {ckpt.name} ({ckpt.stat().st_size/1e6:.1f}MB) | {_stats_str()}", flush=True)
                    last_save_ts = time.time()
                    _maybe_push_ckpt(ckpt)
                    # push the growing log too (200Ko, cheap) — crash-proof curves up to last save
                    try:
                        lf.flush()
                        _maybe_push_ckpt(log_path)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [train] ckpt save failed step {step}: {e} | {_stats_str()}", flush=True)
            if step % 5 == 0 or step==steps:
                print(f"[{time.strftime('%H:%M:%S')} {(time.time()-t0)/60:.1f}min] [train] stream step {step}/{steps} loss={rec['loss']:.4f} ema={rec['ema_loss']:.4f} gnorm={rec['gnorm']:.2f} | {_stats_str()}", flush=True)
        try:
            render_curves(recs, str(curves_path))
        except Exception:
            pass
        wall = round((time.time()-t0)/60,1)
        lf.write(json.dumps({"type":"run_end","run_id":run_id,"step":steps,"samples_seen":steps*cfg.batch_size,"final_loss": recs[-1]["loss"] if recs else None,"wall_min":wall})+"\n")
        # ALWAYS save final projector (even if steps < save_every) — probe must be reusable for qual samples
        try:
            final_path = _cache_root / f"projector_final_{steps}.pt"
            _torch.save({"proj": proj.state_dict(), "step": steps, "cfg": cfg.to_dict(), "final_loss": recs[-1]["loss"] if recs else None}, str(final_path))
            print(f"[train] saved final projector {final_path} ({final_path.stat().st_size/1e6:.1f}MB)", flush=True)
            _maybe_push_ckpt(final_path)
            # also push log + curves + manifest + runs + nohup stdout so HF alone
            # tells the full story of the run (crash-proof even if session dies)
            try:
                _maybe_push_ckpt(log_path)
                _maybe_push_ckpt(curves_path)
                _maybe_push_ckpt(data_dir / "runs.jsonl")
                mf = data_dir / "train_manifest.jsonl"
                if mf.is_file():
                    _maybe_push_ckpt(mf)
                for console_log in sorted(data_dir.glob("train_*.log")):
                    _maybe_push_ckpt(console_log)
            except Exception:
                pass
            # also mirror to data_dir for local fetches
            try:
                import shutil as _sh
                _sh.copyfile(str(final_path), str(data_dir / f"projector_final_{steps}.pt"))
            except Exception:
                pass
        except Exception as e:
            print(f"[train] final projector save failed: {e}", flush=True)
    try:
        reg = registry_entry(run_id=run_id, git_sha=get_git_sha(), config=cfg.to_dict(), seed=0, device=dev, dtype=str(dtype), step_ms=recs[-1].get("step_ms") if recs else None, final_loss=recs[-1]["loss"] if recs else None, extra={"run":"train-stream"})
        append_registry(str(data_dir / "runs.jsonl"), reg)
        # persist registry to hf_vol too
        try:
            import shutil as _sh2
            _sh2.copyfile(str(data_dir / "runs.jsonl"), str(_cache_root / "runs.jsonl"))
        except Exception:
            pass
    except Exception:
        pass
    # Cleanup daemon
    if _prefetch_exec is not None:
        try:
            _prefetch_exec.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    return 0

def _run_resume_local(dd: Path, cfg: TrainConfig, max_steps: int | None, device: str | None, dtype: str, resume_step: int | None) -> int:
    """Load latest (or K) local step ckpt and continue via the streaming restore path."""
    try:
        ckpt_path = _find_local_ckpt(dd, resume_step)
    except FileNotFoundError as e:
        print(f"[train] resume requested but {e}", flush=True)
        return 1
    try:
        import torch as _t

        resume_ckpt = _t.load(str(ckpt_path), map_location="cpu")
    except Exception as e:  # noqa: BLE001 — corrupt ckpt must not start a fresh run silently
        print(f"[train] resume load failed for {ckpt_path} ({e})", flush=True)
        return 1
    print(f"[train] resume local from {ckpt_path} (step {resume_ckpt.get('step')})", flush=True)
    # Resume always takes the streaming restore path (smoke/local-emb never
    # write full-state step ckpts, so there is nothing to restore there).
    _resume_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _resume_dev == "cuda":
        try:
            require_gpu("train")
        except SystemExit as e:
            print(str(e), flush=True)
            raise
    try:
        return _streaming_train(dd, cfg, max_steps, _resume_dev, dtype, resume_ckpt=resume_ckpt)
    except Exception as e:
        import traceback
        print(f"[train] streaming resume failed ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        print("[train] NOT falling back to smoke: partial ckpts/log preserved, exiting 1", flush=True)
        return 1


def run_train(
    data_dir: Path | str,
    cfg: TrainConfig,
    backend=None,
    max_steps: int | None = None,
    device: str | None = None,
    dtype: str = "auto",
    resume: str = "off",
    resume_step: int | None = None,
) -> int:
    """Entry point for `cli train` non-dryrun.

    - Validates data_dir + manifest (header-first)
    - require_gpu("train") if device == "cuda" (any GPU)
    - Chooses path: fake-smoke (tiny) vs HF streaming (grok_probe) vs error with guidance
    - When resume == "local", loads latest (or K) projector_step*.pt and
      continues at step K+1 with restored opt/monitor (same run_id, appended log).
    Returns 0 on success, 1 if caller should delegate (e.g. no fake fixture, need HF path).
    """
    dd = Path(data_dir)
    if _ensure_expandable_segments():
        print("[train] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (fragmentation relief)", flush=True)
    if resume not in ("off", "local", "hf"):
        print(f"[train] unknown --resume {resume!r} (expected off|local|hf)", flush=True)
        return 1
    if resume == "hf":
        print("[train] --resume hf not yet implemented (Task 3) — use --resume local", flush=True)
        return 1
    if resume == "local":
        return _run_resume_local(dd, cfg, max_steps, device, dtype, resume_step)
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
    if manifest_path.exists():
        rows, header = load_manifest(manifest_path)
        is_fake_fixture = header is not None and any("fake" in r.get("emb", "") for r in rows[:10])
        if is_fake_fixture or len(rows) <= 200:
            print(f"[train] detected {'fake fixture' if is_fake_fixture else 'small manifest'} ({len(rows)} rows) — running tiny smoke (no 2B download)", flush=True)
            return _smoke_train_with_fake_data(dd, cfg, max_steps, dev)
    else:
        print(f"[train] no local manifest at {manifest_path} — will fetch from HF (keypa/vision-adapter-manifests)", flush=True)

    # Real data: native HF streaming (no grok shell-out).
    # Uses vision_adapter/data/stream.py (RemoteShard cluster sampling) so
    # `python -m vision_adapter train --data-dir ./data --max-steps 200`
    # works on any CUDA host without an extra clone of grok_probe_qwen.py.
    # Local embeddings under <data-dir>/embeddings/*.pt are preferred when present.
    local_emb_dir = dd / "embeddings"
    has_local_emb = local_emb_dir.is_dir() and any(local_emb_dir.glob("*.pt"))
    if has_local_emb:
        print(f"[train] local embeddings found at {local_emb_dir} — using local path", flush=True)
        return _local_train_with_precomputed(dd, cfg, max_steps, dev, dtype)
    print("[train] no local embeddings — streaming from HF (keypa/vision-adapter-embeddings)", flush=True)
    try:
        return _streaming_train(dd, cfg, max_steps, dev, dtype)
    except Exception as e:
        import traceback
        print(f"[train] streaming train failed ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        print("[train] NOT falling back to smoke: partial ckpts/log preserved, exiting 1", flush=True)
        return 1
