#!/usr/bin/env python3
"""modal_probe.py — Rung 2 of the validation ladder: Qwen3.5-2B grok probe
on Modal, using the PRODUCTION data plane.

Purpose: same recipe, smaller model, real data path. Where the Colab probe
(grok_probe_qwen.py) streamed parquet over HTTP, this reads the cached
MoonViT embeddings straight from the vision-adapter-data Volume via EmbSFT —
byte-identical to what modal_train.py will feed DeepSeek-V4-Flash. A grok
timestamp measured HERE parameterizes the B300 launch.

Reused from production (modal_train.py):
    - EmbSFT dataset (.pt reads from /data/embeddings) — imported, not copied
    - Volume / HF-cache wiring, checkpoint-to-Volume + vol.commit() cadence
    - telemetry contract (flush prints, JSONL per step, curves PNG)

Ported from the VALIDATED Colab probe (grok_probe_qwen.py):
    - HourglassProjector LN(v)->Linear(v,2h)->GELU->Linear(2h,h), h=read
      from Qwen config at runtime (25,180,160 params @ h=2048)
    - inputs_embeds injection at [1:1+n_vis] (Qwen forbids ids+embeds)
    - SELECTIVE lm_head loss: backbone runs without logits; CE applied only
      at supervised positions. Mandatory here too — full-sequence logits
      over Qwen's 248k vocab would be ~80 GiB at bs16.
    - collate with bos=None guard (Qwen3.5 has no BOS; DeepSeek has one,
      so modal_train.make_collate itself stays untouched)
    - plateau banners, >2x-floor spike alerts, median-baseline collapse
      detector, warmup->cosine LR

Precision: bf16 end-to-end (L4/A100 are Ampere+/Ada => native bf16). No
GradScaler, no quantization — mirrors modal_train's projector-in-bf16.

Run (types coerce — Modal delivers CLI params as strings):
    modal run modal_probe.py::dryrun                     # memory gate first
    modal run modal_probe.py::train --max-steps 7500     # ~3h on L4
Options: --gpu l4|a100  --batch-size 16  --sample-size 0(full)|N  --resume
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import deque

import modal
from vision_adapter.config import probe_config as _probe_config, config_header as _config_header  # noqa: E402 — single source, see grok_probe_qwen

# ----------------------------- infra ----------------------------------------

vol = modal.Volume.from_name("vision-adapter-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("vision-adapter-hf", create_if_missing=True)
VOLUME_DIR = "/data"
HF_CACHE = "/hf"

MODEL_REPO = "Qwen/Qwen3.5-2B"
DS_TRAIN_MANIFEST = "train_manifest.jsonl"       # production manifest (120k)
CKPT_DIR_REL = "probe_ckpts_qwen"
LOG_DIR_REL = "probe_logs_qwen"
LOG_FILE = "qwen_probe_log.jsonl"
CURVES_PNG = "qwen_probe_curves.png"
FINAL_PATH = "projector_qwen_final.safetensors"

GPUS = {"l4": ("L4", 22.0),                      # (profile, mem-cap GiB)
        "a100": ("A100-40GB", 34.0)}

# Single source of truth — see vision_adapter/config.py (replaces the 3×
# scattered LR/BATCH_SIZE/MAX_SEQ_LEN/WARMUP at grok_probe:104 /
# modal_probe:65 / modal_train:49). Module-level aliases kept for
# test + intra-file references.
_CFG_PROBE = _probe_config()
LR = _CFG_PROBE.lr
WARMUP_STEPS = _CFG_PROBE.warmup_steps
GRAD_CLIP = _CFG_PROBE.grad_clip
MAX_SEQ_LEN = _CFG_PROBE.max_seq_len
DEFAULT_BS = _CFG_PROBE.batch_size
SAMPLES_PER_BASETEN_GROK = _CFG_PROBE.samples_per_baseten_grok
PLATEAU_CHECK_EVERY = _CFG_PROBE.plateau_check_every
PLATEAU_WINDOW = _CFG_PROBE.plateau_window
PLATEAU_REL_TOL = _CFG_PROBE.plateau_rel_tol
EMA_BETA = _CFG_PROBE.ema_beta
SPIKE_FACTOR = _CFG_PROBE.spike_factor
SPIKE_WINDOW = _CFG_PROBE.spike_window
SPIKE_MIN_HISTORY = _CFG_PROBE.spike_min_history
STATUS_EVERY = _CFG_PROBE.status_every
CHART_EVERY = _CFG_PROBE.chart_every
SAVE_EVERY = _CFG_PROBE.save_every
VAL_PROBE_EVERY = _CFG_PROBE.val_every

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",                # cu130; Ada/Ampere verified stack
        "transformers>=5.12",
        "safetensors", "accelerate", "datasets",
        "pillow", "sentencepiece", "huggingface_hub",
        "matplotlib",
    )
    .env({"HF_HOME": HF_CACHE})
)

app = modal.App("qwen-grok-probe")

# modal_train is imported inside the container for EmbSFT; the modern Modal
# API wants the dependency expressed on the Image (Mount is deprecated).
image = image.add_local_file("modal_train.py", "/root/modal_train.py")
image = image.add_local_dir("vision_adapter", "/root/vision_adapter")

_T0 = time.time()


def _phase(msg: str) -> None:
    print(f"[probe] +{time.time() - _T0:6.1f}s  {msg}", flush=True)


def _as_int(v, default):
    """Modal delivers CLI params as strings (HANDOFF gotcha)."""
    return default if v is None else int(v)


def _as_float(v, default):
    return default if v is None else float(v)


def _as_bool(v):
    return bool(v) and str(v).lower() not in ("0", "false", "no")


# ----------------------------- model pieces ---------------------------------
# (ports of the validated grok_probe_qwen.py core — see module docstring)


class HourglassProjector(modal_cls := __import__("torch").nn.Module):
    def __init__(self, vision_dim: int = 4096, llm_dim: int = 2048):
        super().__init__()
        import torch.nn as nn
        self.ln = nn.LayerNorm(vision_dim)
        self.up = nn.Linear(vision_dim, 2 * llm_dim)
        self.act = nn.GELU()
        self.dn = nn.Linear(2 * llm_dim, llm_dim)

    def forward(self, x):
        return self.dn(self.act(self.up(self.ln(x))))


def load_backbone(limit_layers: int | None, device: str,
                  use_grad_checkpoint: bool = True,
                  attn_impl: str = "eager",
                  use_compile: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _phase(f"tokenizer {MODEL_REPO} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    _phase(f"loading backbone {MODEL_REPO} (bf16, no quantization) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map=device)
    if limit_layers:
        model.model.layers = model.model.layers[:int(limit_layers)]
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    # train() NOT eval(): GradientCheckpointingLayer gates on self.training
    # (same gotcha as modal_train; numerically safe — no stochastic layers)
    model.train()
    if use_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        _phase("gradient checkpointing: ON (recompute to save VRAM)")
    else:
        _phase("gradient checkpointing: OFF (activations retained, faster)")
    # Attn impl flag — FlexAttention is feature-flagged alternative
    if attn_impl == "flex":
        _phase("FlexAttention requested — checking availability ...")
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
            _phase("FlexAttention available (scaffold active, eager fallback until patched)")
            # Full Flex patch (sinks as score_mod, head_dim=512, GQA) lands in Phase 5
        except ImportError:
            print("[probe] WARNING: FlexAttention not available on this torch — falling back to eager",
                  flush=True)
    if use_compile:
        try:
            model = torch.compile(model, fullgraph=False)
            _phase("torch.compile: enabled")
        except Exception as e:
            print(f"[probe] torch.compile failed (ignored): {e}", flush=True)
    cfg = getattr(model.config, "text_config", model.config)
    _phase(f"backbone ready: hidden_size={cfg.hidden_size} "
           f"layers={len(model.model.layers)} | bos={tok.bos_token_id} "
           f"eos={tok.eos_token_id} grad_ckpt={use_grad_checkpoint} attn={attn_impl} compile={use_compile}")
    return tok, model, int(cfg.hidden_size)


def make_collate(tok, pad_id: int, max_len: int = MAX_SEQ_LEN):
    """Port of modal_train.make_collate with the bos=None guard (Qwen3.5 has
    no BOS token). Layout: [BOS?] [img x nv] [user] [answer] [EOS]; answer
    has priority on the text budget; labels only on answer+EOS."""
    import torch

    def collate(batch):
        B = len(batch)
        n_vis = [int(b["vis"].shape[0]) for b in batch]
        max_v = max(n_vis)
        u_ids = [tok(b["user"], add_special_tokens=False)["input_ids"] for b in batch]
        a_ids = [tok(b["assistant"], add_special_tokens=False)["input_ids"] for b in batch]
        seq_lens, parts = [], []
        for u, a, nv in zip(u_ids, a_ids, n_vis):
            budget_text = max_len - nv - 2
            a = a[: max(1, budget_text)]
            u = u[: max(1, budget_text - len(a))]
            parts.append((nv, u, a))
            seq_lens.append(1 + nv + len(u) + len(a) + 1)
        L = max(max_v, max(seq_lens))
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, (nv, u, a) in enumerate(parts):
            if tok.bos_token_id is not None:
                input_ids[i, 0] = tok.bos_token_id
            pos = 1 + nv
            input_ids[i, pos: pos + len(u)] = torch.tensor(u)
            ans_start = pos + len(u)
            input_ids[i, ans_start: ans_start + len(a)] = torch.tensor(a)
            input_ids[i, ans_start + len(a)] = tok.eos_token_id
            attn[i, : ans_start + len(a) + 1] = 1
            labels[i, ans_start: ans_start + len(a) + 1] = \
                input_ids[i, ans_start: ans_start + len(a) + 1]
        vis_pad = torch.zeros(B, max_v, 4096, dtype=torch.float32)
        for i, b in enumerate(batch):
            vis_pad[i, : n_vis[i]] = b["vis"]
        return {"input_ids": input_ids, "attention_mask": attn,
                "labels": labels, "vis": vis_pad,
                "n_vis": torch.tensor(n_vis, dtype=torch.long),
                "g": [b.get("g", "?") for b in batch]}

    return collate


def embeds_for(model, batch, proj, device):
    """inputs_embeds-only injection; visual span overwritten by projected
    cached embeddings (autograd flows to the projector through those rows).

    LayerNorm is the rare op whose layer norm math insists on FP32 even in a
    BF16 model — the vis bytes come in as FP32 (EmbSFT writes .float()), so
    cast into the projector's param dtype before calling it. Same fix the
    Colab path hit on CPU, valid everywhere."""
    import torch
    ids = batch["input_ids"].to(device)
    out_dtype = next(model.parameters()).dtype
    proj_dtype = next(proj.parameters()).dtype
    with torch.no_grad():
        base = model.get_input_embeddings()(ids).to(out_dtype)
    vis = batch["vis"].to(device).to(proj_dtype)           # F32 -> projector dtype
    pv = proj(vis).to(out_dtype)                           # then into model dtype
    merged = base.clone()
    for i, nv in enumerate(batch["n_vis"].tolist()):
        merged[i, 1: 1 + nv] = pv[i, :nv]
    return {"inputs_embeds": merged,
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device)}


def train_step(model, proj, opt, batch, device, clip: float = GRAD_CLIP):
    """fwd/bwd/clip/step with SELECTIVE lm_head loss (see module docstring).
    Loss is mathematically identical to model(labels=...) up to fp order."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    t0 = time.time()
    inp = embeds_for(model, batch, proj, device)
    labels = inp.pop("labels")
    base = model.model                              # text stack, no lm_head
    out = base(inputs_embeds=inp["inputs_embeds"],
               attention_mask=inp["attention_mask"])
    hidden = out.last_hidden_state
    shift = labels[:, 1:]
    mask = shift != -100
    pos = mask.nonzero(as_tuple=False)
    h_sel = hidden[:, :-1][pos[:, 0], pos[:, 1]]
    y_sel = shift[pos[:, 0], pos[:, 1]]
    logits_sel = model.lm_head(h_sel).float()       # [N, V] — tiny
    loss = F.cross_entropy(logits_sel, y_sel)

    finite = bool(torch.isfinite(loss))
    opt.zero_grad(set_to_none=True)
    if finite:
        loss.backward()
        gnorm = float(nn.utils.clip_grad_norm_(proj.parameters(), clip))
        if math.isfinite(gnorm):
            opt.step()
        else:
            finite = False
    else:
        gnorm = float("nan")
    return {"loss": float(loss.item()), "finite": finite, "gnorm": gnorm,
            "tokens": int(batch["attention_mask"].sum()),
            "batch_size": int(batch["input_ids"].shape[0]),
            "step_ms": round((time.time() - t0) * 1000, 1)}


def lr_at(step: int, max_steps: int, base_lr: float = LR,
          warmup: int = WARMUP_STEPS) -> float:
    if step <= warmup:
        return base_lr * max(1, step) / warmup
    t = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


# ----------------------------- telemetry ------------------------------------


class ProbeMonitor:
    """EMA stream + flat-loss plateau banners + EMA-vs-recent-floor spikes +
    median-baseline collapse detector (ports of the validated versions)."""

    def __init__(self):
        self.ema = None
        self.history = deque(maxlen=PLATEAU_WINDOW)
        self.ema_history = deque(maxlen=SPIKE_WINDOW)
        self.last_banner_step = -(10 ** 9)
        self.last_alert_step = -(10 ** 9)
        self.n_alerts = 0
        self.n_banners = 0
        self.collapse_step = None

    def update(self, step, loss, samples_seen):
        self.ema = loss if self.ema is None else \
            EMA_BETA * self.ema + (1 - EMA_BETA) * loss
        self.history.append(loss)
        self.ema_history.append(self.ema)

        if self.collapse_step is None and len(self.history) >= PLATEAU_WINDOW:
            base = statistics.median(self.history)
            recent = sum(list(self.history)[-PLATEAU_CHECK_EVERY:]) / PLATEAU_CHECK_EVERY
            if recent < 0.5 * base:
                self.collapse_step = step
                print(f"\n*** COLLAPSE *** step={step} "
                      f"samples_seen={samples_seen} (recent-mean {recent:.4f} "
                      f"< 0.5 * window-median {base:.4f})\n", flush=True)

        if (len(self.ema_history) >= SPIKE_MIN_HISTORY
                and step - self.last_alert_step > 50
                and self.ema > SPIKE_FACTOR * min(self.ema_history)):
            print(f"[SPIKE-ALERT] step {step}: ema {self.ema:.4f} > "
                  f"{SPIKE_FACTOR:.0f}x recent floor "
                  f"{min(self.ema_history):.4f}", flush=True)
            self.last_alert_step = step
            self.n_alerts += 1

        if (step - self.last_banner_step >= PLATEAU_CHECK_EVERY
                and len(self.history) >= PLATEAU_WINDOW):
            old = self.history[0]
            if old > 0 and abs(self.ema - old) / old < PLATEAU_REL_TOL:
                print("\n" + "=" * 74 + "\n"
                      "[plateau] FLAT LOSS IS EXPECTED DURING THE GROK PHASE.\n"
                      f"[plateau] reference collapse ~{SAMPLES_PER_BASETEN_GROK}"
                      " samples_seen\n"
                      f"[plateau] current samples_seen={samples_seen} "
                      f"({100 * samples_seen / SAMPLES_PER_BASETEN_GROK:.1f}% "
                      "of reference)\n"
                      "[plateau] keep going.\n" + "=" * 74 + "\n", flush=True)
                self.last_banner_step = step
                self.n_banners += 1
        return self.ema


def render_curves(records, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[probe] chart import failed (ignored): {e}", flush=True)
        return False
    tr = [r for r in records if r.get("type") == "train"]
    if not tr:
        return False
    xs = [r["step"] for r in tr]
    ss = [r["samples_seen"] for r in tr]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    ax = axes[0]
    ax.plot(xs, [r["loss"] for r in tr], lw=0.4, alpha=0.35,
            color="tab:blue", label="train loss (raw)")
    ax.plot(xs, [r["ema_loss"] for r in tr], lw=1.8, color="tab:blue",
            label="EMA(.98)")
    ax.plot(xs, [r["lr"] for r in tr], lw=1.0, color="tab:purple", label="lr")
    ax.set_yscale("log")
    ax.set_ylabel("loss (log) / lr")
    ax.set_xlabel("steps")
    ax.set_title(f"Qwen3.5-2B grok probe — {len(tr)} steps | ref ~"
                 f"{SAMPLES_PER_BASETEN_GROK} samples")
    ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.secondary_xaxis("top", functions=(
        lambda s: s * ss[-1] / max(1, xs[-1]),
        lambda v: v * max(1, xs[-1]) / max(1, ss[-1])))
    ax2.set_xlabel("samples_seen")

    ax = axes[1]
    ax.plot(xs, [r["gnorm"] for r in tr], lw=0.5, color="tab:green",
            label="grad norm (pre-clip)")
    ax.set_yscale("log")
    ax.set_xlabel("steps")
    ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.secondary_xaxis("top", functions=(
        lambda s: s * ss[-1] / max(1, xs[-1]),
        lambda v: v * max(1, xs[-1]) / max(1, ss[-1])))
    ax2.set_xlabel("samples_seen")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


# ============================ shared setup ===================================


def _vis_len_of_entry(entry) -> int:
    import torch
    try:
        t = torch.load(entry["emb_abs"], map_location="cpu", weights_only=True)
        return int(t.shape[0])
    except Exception:
        return 0


def _shared_setup(batch_size: int, sample_size: int, limit_layers,
                  use_grad_checkpoint: bool = True,
                  bucketing: bool = False,
                  attn_impl: str = "eager",
                  use_compile: bool = False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model, llm_dim = load_backbone(
        limit_layers, device={"": 0},
        use_grad_checkpoint=use_grad_checkpoint,
        attn_impl=attn_impl,
        use_compile=use_compile)
    proj = HourglassProjector(4096, llm_dim).to(device="cuda",
                                                dtype=torch.bfloat16)
    for p in proj.parameters():
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"[probe] HourglassProjector params={n_params:,} | bf16", flush=True)

    # PRODUCTION data plane: EmbSFT reading /data/embeddings/*.pt (imported
    # from modal_train so the contract cannot drift from the DeepSeek run).
    from modal_train import EmbSFT
    ds = EmbSFT(vol_dir=VOLUME_DIR, manifest_rel=DS_TRAIN_MANIFEST)
    if sample_size and 0 < sample_size < len(ds):
        import random
        rng = random.Random(17)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        ds.rows = [ds.rows[i] for i in idx[:sample_size]]
        print(f"[probe] subsampled to {len(ds)} rows (seed 17)", flush=True)
    # A2 bucketing: group by N_vis into bins to cut padding waste
    # Bins mirror grok_probe_qwen.py's validated scheme: <500, 500-1500, 1500-3000, >3000
    bucket_edges = bucketing if isinstance(bucketing, list) else ([500, 1500, 3000] if bucketing else None)
    # Dataset is Indexed; we expose rows directly — bucketing reorders the row list
    if bucket_edges is not None:
        # Use cheap N_vis already available via .pt header where possible;
        # fallback to scanning manifest length. We bucket ROWS, not batches.
        _phase(f"bucketing N_vis into bins {bucket_edges} ...")
        buckets: dict[int, list] = {}
        for r in ds.rows:
            nv = _vis_len_of_entry(r)
            bin_id = sum(nv >= e for e in bucket_edges)
            buckets.setdefault(bin_id, []).append(r)
        # Reassemble: bucket-major order (like stream_order), shuffled within bucket
        import random as _rng
        rng2 = _rng.Random(17)
        ordered = []
        for bid in sorted(buckets):
            grp = buckets[bid]
            rng2.shuffle(grp)
            ordered.extend(grp)
            print(f"  bucket {bid}: {len(grp)} rows", flush=True)
        ds.rows = ordered
        _phase(f"bucketing done: {len(ds.rows)} rows across {len(buckets)} buckets")
    collate = make_collate(tok, tok.pad_token_id)
    # Profiling-ready loader: keep production defaults; Phase-2 ladder measures
    # with cuda events rather than wall-clock in train_step
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=8, persistent_workers=True,
        pin_memory=True)                    # production loader settings
    _phase(f"dataset ready: {len(ds)} rows from the Volume "
           f"(warm .pt reads, production path) | grad_ckpt={use_grad_checkpoint} "
           f"bucketing={bool(bucketing)} attn={attn_impl} compile={use_compile}")
    opt = torch.optim.AdamW(proj.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.0)
    return tok, model, proj, opt, loader


# ============================== dryrun gate ==================================


@app.function(image=image, gpu="L4",
              volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=3600, memory=32768)
def dryrun(batch_size: int = DEFAULT_BS,
           no_grad_ckpt: bool = False,
           bucketing: bool = False,
           attn: str = "eager",
           compile: bool = False):
    """One fwd/bwd at the target batch size; asserts peak VRAM under the gate
    and records the step-time baseline (mirrors modal_train.train_dryrun).

    Ladder flags (each maps to one column in the measured table):
      no_grad_ckpt  -> A3: disable gradient checkpointing
      bucketing     -> A2: visual-token-length bucketing
      attn flex     -> A1: FlexAttention feature-flagged alternative
      compile       -> torch.compile on top of the chosen attn impl
    """
    import torch
    props = torch.cuda.get_device_properties(0)
    print(f"[dryrun] gpu={props.name} cc={props.major}.{props.minor} "
          f"vram={props.total_memory / 2**30:.0f}GiB | "
          f"no_grad_ckpt={_as_bool(no_grad_ckpt)} bucketing={_as_bool(bucketing)} "
          f"attn={attn} compile={_as_bool(compile)}", flush=True)
    tok, model, proj, opt, loader = _shared_setup(
        _as_int(batch_size, DEFAULT_BS), 0, None,
        use_grad_checkpoint=not _as_bool(no_grad_ckpt),
        bucketing=_as_bool(bucketing),
        attn_impl=str(attn), use_compile=_as_bool(compile))
    sig = next(iter(loader))
    out = train_step(model, proj, opt, sig, "cuda")
    peak = torch.cuda.max_memory_allocated() / 2**30
    line = (f"[dryrun] loss={out['loss']:.4f} peak={peak:.2f}GiB "
            f"step_ms={out['step_ms']} bs={sig['input_ids'].shape[0]} | "
            f"ckpt={'off' if _as_bool(no_grad_ckpt) else 'on'} "
            f"bucketing={_as_bool(bucketing)} attn={attn}")
    print(line, flush=True)
    assert peak < 22.0, f"MEMORY GATE FAIL ({peak:.2f} GiB) — lower batch size"
    print("[dryrun] MEMORY GATE PASS", flush=True)


# ================================ train ======================================


def _instrumented_train_step(model, proj, opt, batch, device):
    """Same math as train_step but with CUDA-event phase breakdown.

    Phases: projector_fwd, llm_fwd, loss, llm_bwd, projector_bwd+optim.
    Uses CUDA events when available, falls back to perf_counter for CPU.
    train_step timing is wall-clock cpu; this adds device-side splits.
    """
    import torch
    try:
        _probe = torch.cuda.Event(enable_timing=True)
        use_cuda_events = True
        del _probe
    except Exception:
        use_cuda_events = False

    def _elapsed(s, e):
        if use_cuda_events:
            torch.cuda.synchronize()
            return s.elapsed_time(e)
        return 0.0

    import torch.nn.functional as F
    import torch.nn as nn
    import math as _math
    t0 = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    if use_cuda_events:
        t0.record()
    else:
        import time as _time
        wall0 = _time.perf_counter()

    # Embed + projector
    t_proj0 = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    if use_cuda_events:
        t_proj0.record()
    inp = embeds_for(model, batch, proj, device)
    if use_cuda_events:
        t_proj1 = torch.cuda.Event(enable_timing=True)
        t_proj1.record()

    labels = inp.pop("labels")
    base = model.model
    llm_fwd0 = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    if use_cuda_events:
        llm_fwd0.record()
    out = base(inputs_embeds=inp["inputs_embeds"],
               attention_mask=inp["attention_mask"])
    hidden = out.last_hidden_state
    if use_cuda_events:
        llm_fwd1 = torch.cuda.Event(enable_timing=True)
        llm_fwd1.record()

    shift = labels[:, 1:]
    mask = shift != -100
    pos = mask.nonzero(as_tuple=False)
    h_sel = hidden[:, :-1][pos[:, 0], pos[:, 1]]
    y_sel = shift[pos[:, 0], pos[:, 1]]
    logits_sel = model.lm_head(h_sel).float()
    loss = F.cross_entropy(logits_sel, y_sel)
    loss_ms = 0.0  # ce is microseconds vs ms-scale llm
    if use_cuda_events:
        loss_ev = torch.cuda.Event(enable_timing=True)
        loss_ev.record()

    finite = bool(torch.isfinite(loss))
    opt.zero_grad(set_to_none=True)
    bwd0 = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    if use_cuda_events:
        bwd0.record()
    if finite:
        loss.backward()
        gnorm = float(nn.utils.clip_grad_norm_(proj.parameters(), 1.0))
        if _math.isfinite(gnorm):
            opt.step()
        else:
            finite = False
    else:
        gnorm = float("nan")
    if use_cuda_events:
        bwd1 = torch.cuda.Event(enable_timing=True)
        bwd1.record()
        torch.cuda.synchronize()
        proj_fwd_ms = t_proj0.elapsed_time(t_proj1) if use_cuda_events else 0
        llm_fwd_ms = llm_fwd0.elapsed_time(llm_fwd1) if use_cuda_events else 0
        bwd_ms = bwd0.elapsed_time(bwd1) if use_cuda_events else 0
        total_ms = t0.elapsed_time(bwd1) if use_cuda_events else ( (_time.perf_counter() - wall0)*1000 )
    else:
        proj_fwd_ms = llm_fwd_ms = bwd_ms = total_ms = 0.0
        total_ms = (_time.perf_counter() - wall0)*1000

    return {"loss": float(loss.item()), "finite": finite, "gnorm": gnorm,
            "tokens": int(batch["attention_mask"].sum()),
            "batch_size": int(batch["input_ids"].shape[0]),
            "step_ms": round(total_ms, 1),
            "proj_fwd_ms": round(proj_fwd_ms, 1),
            "llm_fwd_ms": round(llm_fwd_ms, 1),
            "loss_ms": round(loss_ms, 1),
            "bwd_ms": round(bwd_ms, 1)}


@app.function(image=image, gpu="L4",
              volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=43200, memory=32768)
def train(batch_size: int = DEFAULT_BS,
          max_steps: int = 7500,
          epochs: int = 1,
          sample_size: int = 0,        # 0 = full 120k production manifest
          resume: bool = False,
          limit_layers: int | None = None,
          gpu_choice: str = "l4",
          no_grad_ckpt: bool = False,
          bucketing: bool = False,
          attn: str = "eager",
          compile: bool = False,
          profile: bool = False):
    import torch
    bs = _as_int(batch_size, DEFAULT_BS)
    want_gpu, _cap = GPUS.get(str(gpu_choice), GPUS["l4"])
    if want_gpu != "L4":
        print(f"[probe] NOTE: restart with --gpu {gpu_choice} mapped at deploy "
              f"time; this function profiled for L4", flush=True)

    max_steps = _as_int(max_steps, 7500)
    epochs = _as_int(epochs, 1)
    sample_size = _as_int(sample_size, 0)
    limit_layers = _as_int(limit_layers, None) if limit_layers else None
    use_profile = _as_bool(profile)

    tok, model, proj, opt, loader = _shared_setup(
        bs, sample_size, limit_layers,
        use_grad_checkpoint=not _as_bool(no_grad_ckpt),
        bucketing=_as_bool(bucketing),
        attn_impl=str(attn), use_compile=_as_bool(compile))
    monitor = ProbeMonitor()
    records: list[dict] = []

    ckpt_dir = os.path.join(VOLUME_DIR, CKPT_DIR_REL)
    log_dir = os.path.join(VOLUME_DIR, LOG_DIR_REL)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    latest_sd = os.path.join(ckpt_dir, "latest.safetensors")
    latest_opt = os.path.join(ckpt_dir, "latest.opt.pt")

    steps_per_epoch = len(loader)
    steps_total = min(max_steps, epochs * steps_per_epoch)
    print(f"[probe] steps/epoch={steps_per_epoch} @bs{bs} | planned="
          f"{steps_total} | grok ref ~{SAMPLES_PER_BASETEN_GROK} samples "
          f"(~step {SAMPLES_PER_BASETEN_GROK // bs})", flush=True)

    start_step = 0
    if _as_bool(resume) and os.path.exists(latest_opt):
        st = torch.load(latest_opt, map_location="cuda", weights_only=False)
        from safetensors.torch import load_file
        proj.load_state_dict({k: v.to("cuda")
                              for k, v in load_file(latest_sd).items()})
        opt.load_state_dict({k: (v.to("cuda") if torch.is_tensor(v) else v)
                             for k, v in st["opt"].items()})
        start_step = int(st["step"])
        if os.path.exists(os.path.join(log_dir, LOG_FILE)):
            with open(os.path.join(log_dir, LOG_FILE)) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get("type") == "train":
                            records.append(r)
                    except json.JSONDecodeError:
                        pass
        if records:
            monitor.ema = records[-1]["ema_loss"]
        # NOTE: shuffle=True means restarted batches differ from an uninterrupted
        # run — same resume semantics as modal_train (weights/opt/step restore;
        # data order is not part of the contract).
        print(f"[probe] RESUME from completed step {start_step}", flush=True)
    for g in opt.param_groups:
        g["lr"] = lr_at(start_step + 1, steps_total)

    logger = open(os.path.join(log_dir, LOG_FILE),
                  "a" if start_step else "w", buffering=1)
    if not start_step:
        try:
            hdr = _config_header(_CFG_PROBE, manifest_path=None,
                                 extra={"type": "config_header", "run": "modal_probe",
                                        "args": {"batch_size": bs, "max_steps": max_steps,
                                                 "sample_size": sample_size, "epochs": epochs,
                                                 "limit_layers": limit_layers, "gpu": want_gpu,
                                                 "grad_ckpt": not _as_bool(no_grad_ckpt),
                                                 "bucketing": _as_bool(bucketing),
                                                 "attn": str(attn), "compile": _as_bool(compile)}})
            logger.write(json.dumps(hdr) + "\n")
        except Exception as e:
            print(f"[probe] WARNING: could not write config header ({e})", flush=True)
    t0 = time.time()
    step, samples_seen, tokens_total = start_step, start_step * bs, 0
    consecutive_bad = 0
    it = iter(loader)

    # Choose step function: profile adds CUDA-event phase split
    _step_fn = _instrumented_train_step if use_profile else train_step
    if use_profile:
        _phase("profiling ON — per-phase CUDA timings will be logged")

    while step < steps_total:
        # data-load timing (H2D happens inside collate/pin_memory overlap)
        import time as _t
        t_data0 = _t.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break                                   # epochs exhausted
        data_ms = (_t.perf_counter() - t_data0) * 1000
        step += 1
        out = _step_fn(model, proj, opt, batch, "cuda")
        if not out["finite"]:
            consecutive_bad += 1
            print(f"[probe][WARN] skipped step {step} "
                  f"(loss={out['loss']:.4f}; {consecutive_bad} consecutive)",
                  flush=True)
            if consecutive_bad >= 5:
                print("[probe][FATAL] 5 consecutive bad steps", flush=True)
                logger.close()
                raise SystemExit(1)
            continue
        consecutive_bad = 0
        samples_seen += out["batch_size"]
        tokens_total += out["tokens"]
        elapsed = max(1e-9, time.time() - t0)
        rec = {"type": "train", "step": step,
               "epoch": step // max(1, steps_per_epoch),
               "loss": round(out["loss"], 5),
               "gnorm": round(out["gnorm"], 4),
               "lr": float(opt.param_groups[0]["lr"]),
               "tokens": out["tokens"], "samples_seen": samples_seen,
               "ts": round(time.time(), 1)}
        ema = monitor.update(step, rec["loss"], samples_seen)
        rec["ema_loss"] = round(ema, 5)
        rec["tok_s"] = round(tokens_total / elapsed, 1)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["it_s"] = round((step - start_step) / elapsed, 3)
        rec["step_ms"] = out["step_ms"]
        if use_profile:
            rec["proj_fwd_ms"] = out.get("proj_fwd_ms", 0.0)
            rec["llm_fwd_ms"] = out.get("llm_fwd_ms", 0.0)
            rec["bwd_ms"] = out.get("bwd_ms", 0.0)
            rec["data_ms"] = round(data_ms, 1)
        records.append(rec)
        logger.write(json.dumps(rec) + "\n")

        if step % STATUS_EVERY == 0 or step == steps_total:
            eta_min = (steps_total - step) * rec["step_ms"] / 1000 / 60
            print(f"[probe] step {step}/{steps_total} loss={rec['loss']:.4f} "
                  f"ema={rec['ema_loss']:.4f} gnorm={rec['gnorm']:.2f} "
                  f"tok/s={rec['tok_s']:.0f} samples_seen={samples_seen} "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB "
                  f"{rec['it_s']:.2f}it/s ETA={eta_min:.0f}m"
                  + (f" ALERTS={monitor.n_alerts}" if monitor.n_alerts else ""),
                  flush=True)
        if step % CHART_EVERY == 0 or step == steps_total:
            render_curves(records, os.path.join(log_dir, CURVES_PNG))
            vol.commit()                            # mid-run fetchable
        if step % SAVE_EVERY == 0 or step == steps_total:
            from safetensors.torch import save_file
            save_file({k: v.detach().cpu().contiguous()
                       for k, v in proj.state_dict().items()}, latest_sd)
            torch.save({"opt": opt.state_dict(), "step": step}, latest_opt)
            vol.commit()
            print(f"[probe] checkpoint saved @ step {step}", flush=True)

    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous()
               for k, v in proj.state_dict().items()},
              os.path.join(VOLUME_DIR, FINAL_PATH))
    render_curves(records, os.path.join(log_dir, CURVES_PNG))
    final_loss = records[-1]["loss"] if records else float("nan")
    logger.write(json.dumps({
        "type": "run_end", "step": step, "samples_seen": samples_seen,
        "final_loss": final_loss, "final_ema": monitor.ema,
        "collapse_step": monitor.collapse_step,
        "collapse_samples_seen": (monitor.collapse_step or 0) * bs,
        "n_alerts": monitor.n_alerts, "n_banners": monitor.n_banners,
        "wall_min": round((time.time() - t0) / 60, 1)}) + "\n")
    logger.close()
    vol.commit()
    print(f"[probe] DONE step={step} samples_seen={samples_seen} "
          f"final_loss={final_loss} ema={monitor.ema} "
          f"collapse_at_samples_seen={monitor.collapse_step and monitor.collapse_step * bs}",
          flush=True)
