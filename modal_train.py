#!/usr/bin/env python3
"""
modal_train.py — Phase 4 trainer: graft a MoonViT-V2 vision adapter onto the
frozen DeepSeek-V4-Flash-0731 text backbone, using PRECOMPUTED image embeddings.

Memory model (user-approved): 1× A100-80GB, DeepSeek stays natively-quantised
(~155 GiB on disk) and CPU-offloaded via accelerate device_map="auto" with a
70 GiB GPU cap; vision embeddings are precomputed (MoonViT frozen) so the ViT
never runs in the training hot loop. Only the 67.1M projector is trainable.

Contract: sequence layout per example
    [BOS] [img_emb × n_vis] [user text tokens] [answer tokens] [EOS]
  - the n_vis vision embeddings are spliced into embed_tokens' OUTPUT at
    positions [1 : 1+n_vis] via a forward hook (input_ids stay the model
    input: V4's hash-MoE gate routes by tid2eid[input_ids] and forbids
    inputs_embeds); attention_mask covers them with 1s;
  - labels: -100 everywhere EXCEPT the answer + EOS token positions.

Stages:
  train_dryrun : ONE fwd/bwd at batch=8; asserts peak VRAM < 70 GiB; else ABORT.
  train        : full SFT loop (LR 5e-4, bs 8, adamw) once dry run passes.

Run:    modal run modal_train.py::train_dryrun     # then
        modal run modal_train.py::train
"""
from __future__ import annotations
import os, json, glob, time
import torch  # module-level: EmbSFT(torch.utils.data.Dataset) below needs it at import
import modal

GPU = "A100-80GB"
GPU_MEM_CAP_GIB = 70.0

# A100 (cc 8.0) cannot run FP8 kernels: transformers dequantizes the FP8
# checkpoint (~155 GiB on disk) to bf16 at load. Measured twice (ap-7aS2E…,
# ap-shpUl…): a 70+280 GiB budget still trips accelerate's disk tier, so the
# real accounting is well above 350 GiB once per-layer placement granularity,
# scales, and loader slack are counted. Give it 70+400 and a container well
# above the hint; offload_folder absorbs any marginal spill instead of crashing.
SYS_RAM_CAP_GIB = 400
A100_CONTAINER_RAM_GB = 480

# ---- B300 stack (Phase 5) — 288 GiB VRAM: whole quantized backbone in VRAM ----
B300_GPU = "B300"
B300_GPU_MEM_CAP_GIB = 250.0          # dryrun gate; activations headroom verified, not assumed
TORCH_B300_PIN = "torch==2.13.0"      # cu130 build; SM103 support validated by the cc print in the b300 dryrun
B300_CONTAINER_RAM_GB = 200           # all-in-VRAM path: RAM only feeds the shard-by-shard loader
BATCH_SIZE = 8
LR = 5e-4
MAX_SEQ_LEN = 4096
EPOCHS = 2

# Telemetry / grok-window bookkeeping. Baseten's GLM recipe grokked at ~step 900
# of 1035 (1 epoch @ batch 64 over 66k imgs) => ~58k samples seen. We have 120k
# samples and must watch the loss curve for the same cliff, expected somewhere
# around 7-11k *our* steps at batch 8 (equivalent samples-seen), not at step 900.
SAMPLES_PER_BASETEN_GROK = int(900 * 64)
LOG_EVERY = 1
VAL_EVERY = 250
SAVE_EVERY = 200
# Optional: stop after the curve has collapsed. Off by default; the empirically
# safe move per Baseten is to let the full scheduled epochs run.
GROK_STOP_AFTER_STEPS = None

VOL_NAME = "vision-adapter-data"
EMB_ROOT_REL = "embeddings"
CKPT_DIR_REL = "checkpoints"
DATASET_MANIFEST_REL = "train_manifest.jsonl"  # produced by Phase3 etl+mix stage
VAL_MANIFEST_REL = "train_manifest_val.jsonl"  # held-out split (same build fn)
LOG_DIR_REL = "logs"
LOG_FILE_REL = "train_log.jsonl"

DS_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
MOONVIT_REPO = "keypa/MoonViT-V2-Standalone"

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name("vision-adapter-hf", create_if_missing=True)
VOLUME_DIR = "/data"
HF_CACHE = "/hf"

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "safetensors", "accelerate", "transformers",
        "datasets", "pillow", "sentencepiece", "huggingface_hub",
        "matplotlib",
    )
    .env({"HF_HOME": HF_CACHE})
)

train_image_b300 = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        TORCH_B300_PIN, "safetensors", "accelerate", "transformers",
        # finegrained-fp8: keeps DeepSeek-V4 FP8 blocks resident (no bf16
        # dequant); transformers loads its GEMM kernel through this at runtime.
        "kernels>=0.16.0,<0.17",
        "datasets", "pillow", "sentencepiece", "huggingface_hub",
        "matplotlib",
    )
    .env({"HF_HOME": HF_CACHE})
)

app = modal.App("vision-adapter-train")


# ============================ model assembly (shared) =========================

def build_model(offload: bool = True):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[train] +{time.time() - _T0:6.1f}s  tokenizer: {DS_REPO}", flush=True)
    tok = AutoTokenizer.from_pretrained(DS_REPO, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"[train] +{time.time() - _T0:6.1f}s  loading backbone (~155 GiB) — "
          f"this is the long part, minutes on a cold container ...", flush=True)
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if offload:
        # A100 path: quantized backbone split across GPU cap + CPU RAM (PCIe hops in the loop).
        load_kwargs.update(device_map="auto",
                           max_memory={0: f"{int(GPU_MEM_CAP_GIB)}GiB",
                                       "cpu": f"{int(SYS_RAM_CAP_GIB)}GiB"},
                           offload_folder="/root/offload")
    else:
        # B300 path: 288 GiB VRAM holds the whole model — no offload, no PCIe bottleneck.
        load_kwargs.update(device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained(DS_REPO, **load_kwargs)
    _patch_chunked_eager_attention()   # eager attn OOMs at bs=8 next to the FP8 backbone
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    # train(), NOT eval(): GradientCheckpointingLayer gates on `self.training`
    # (modeling_layers.py L80) — eval() silently disables checkpointing and all
    # 43 layers' activations are retained => OOM. Safe: attention_dropout=0.0
    # and the arch has no other stochastic layers, so numerics are identical.
    model.train()

    class HourglassProjector(nn.Module):
        def __init__(s, d=4096, h=8192):
            super().__init__()
            s.ln = nn.LayerNorm(d)
            s.up = nn.Linear(d, h)
            s.act = nn.GELU()
            s.dn = nn.Linear(h, d)

        def forward(s, x):
            return s.dn(s.act(s.up(s.ln(x))))

    proj = HourglassProjector()
    import torch as _t
    proj = proj.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    for p in proj.parameters():
        p.requires_grad_(True)
    return tok, model, proj


# ============================== dataset / collator ============================

class EmbSFT(torch.utils.data.Dataset):
    """Each row: {emb: <vol-relative .pt>, user: str, assistant: str}.

    The embedding file contains the MoonViT-V2 merged tokens [n_vis, 4096]
    (BF16) already projected by the (unchanged-by-us) Kimi merge; the TRAINABLE
    projector produces the final LLM embedding for each token.

    Falls back silently to an empty dataset when the manifest doesn't exist yet
    (so the optional val split can be absent without breaking training).
    """

    def __init__(self, vol_dir=VOLUME_DIR, manifest_rel=DATASET_MANIFEST_REL):
        self.vol = vol_dir
        self.rows = []
        manifest_path = os.path.join(vol_dir, manifest_rel)
        if not os.path.exists(manifest_path):
            return
        with open(manifest_path) as f:
            for line in f:
                r = json.loads(line)
                r["emb_abs"] = os.path.join(vol_dir, r["emb"])
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        r = self.rows[i]
        vis = torch.load(r["emb_abs"], map_location="cpu").float()   # [n_vis,4096]
        return {"vis": vis, "user": r["user"], "assistant": r["assistant"], "g": r.get("g", "?")}


def make_collate(tok, pad_id, max_len=MAX_SEQ_LEN):
    def collate(batch):
        import torch
        B = len(batch)
        n_vis = [int(b["vis"].shape[0]) for b in batch]
        max_v = max(n_vis)
        # user/answer token ids (image is injected separately, not vocab tokens)
        u_ids = [tok(b["user"], add_special_tokens=False)["input_ids"] for b in batch]
        a_ids = [tok(b["assistant"], add_special_tokens=False)["input_ids"] for b in batch]
        # text budget: reserve [1 BOS] + [n_vis img] + [1 EOS]
        seq_lens = []
        parts = []
        for u, a, nv in zip(u_ids, a_ids, n_vis):
            budget_text = max_len - nv - 2
            # answer has PRIORITY on the text budget (it is the loss target);
            # the user prompt absorbs whatever room is left.
            a = a[: max(1, budget_text)]
            u = u[: max(1, budget_text - len(a))]
            parts.append((nv, u, a))
            seq_lens.append(1 + nv + len(u) + len(a) + 1)
        L = max(max_v, max(seq_lens))
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        device_flag = dict()
        for i, (nv, u, a) in enumerate(parts):
            input_ids[i, 0] = tok.bos_token_id
            # positions [1 : 1+nv] are IMAGE EMBEDDINGS (injected); their id value unused
            pos = 1 + nv
            input_ids[i, pos: pos + len(u)] = torch.tensor(u)
            ans_start = pos + len(u)
            input_ids[i, ans_start: ans_start + len(a)] = torch.tensor(a)
            input_ids[i, ans_start + len(a)] = tok.eos_token_id
            # attention: 1 for bos..eos (incl. the img span); 0 in right-pad
            attn[i, : ans_start + len(a) + 1] = 1
            # labels: only the answer + EOS contribute to loss
            labels[i, ans_start: ans_start + len(a) + 1] = input_ids[i, ans_start: ans_start + len(a) + 1]
            device_flag[i] = nv
        # vis tokens padded to max_v
        vis_pad = torch.zeros(B, max_v, 4096, dtype=torch.float32)
        for i, b in enumerate(batch):
            vis_pad[i, : n_vis[i]] = b["vis"]
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "vis": vis_pad,
            "n_vis": torch.tensor(n_vis, dtype=torch.long),
            "g": [b["g"] for b in batch],
        }
    return collate


class visual_inject:
    """Project cached ViT embeddings -> LLM dim and splice them at positions
    [1 : 1+n_vis] of the sequence.

    DeepSeek-V4's hash-MoE gates pick experts via tid2eid[input_ids] and the
    core model raises if input_ids AND inputs_embeds are both given, so plain
    inputs_embeds injection cannot work. Instead we keep input_ids as the
    model-facing input and hook embed_tokens, overwriting its OUTPUT rows at
    the visual positions. Gradients reach the projector through the spliced
    rows; text rows stay ordinary frozen-embedding lookups. Usage:

        with visual_inject(sig, proj, model) as full:
            out = model(**full)
    """

    def __init__(self, inputs, proj, model):
        import torch
        embed = model.get_input_embeddings()
        self._embed = embed
        self._dev = next(embed.parameters()).device  # not hardcoded cuda: works sharded/cpu too
        self._vis = proj(inputs["vis"].to(self._dev, torch.bfloat16))  # [B,maxv,4096]
        self._n_vis = [int(n) for n in inputs["n_vis"].tolist()]
        self.model_inputs = {
            "input_ids": inputs["input_ids"].to(self._dev),
            "attention_mask": inputs["attention_mask"].to(self._dev),
            "labels": inputs["labels"].to(self._dev),
        }
        self._handle = None

    def __enter__(self):
        vis, n_vis = self._vis, self._n_vis

        def _splice(module, args, output):
            merged = output.clone()          # [B,L,H]; keep autograd link to vis
            for i, nv in enumerate(n_vis):
                merged[i, 1: 1 + nv] = vis[i, :nv]
                # visual slots are excluded from loss upstream (labels -100)
            return merged

        self._handle = self._embed.register_forward_hook(_splice)
        return self.model_inputs

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def _make_chunked_eager(budget_elems: int = 2 ** 26):
    """Chunked replacement for V4's eager_attention_forward.

    The eager path materializes [.., S_q, S_kv+1] logits in fp32 (~46 GiB at
    bs=8 next to the 155 GiB FP8 backbone => OOM). Softmax is independent per
    query row, so evaluating query chunks is mathematically identical with
    peak transient ~budget_elems. The KV repeat_kv expansion is HOISTED out
    of the chunk loop (calling orig per chunk would re-materialize the
    expanded KV — multiple GiB — once per chunk). Returns (output, None):
    attn_weights are dropped because output_attentions is always False in
    this trainer."""
    import torch as _torch
    import torch.nn.functional as _F

    def _repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        b, h, s, d = x.shape
        return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)

    def chunked(module, query, key, value, attention_mask=None, scaling=None,
                dropout: float = 0.0, **kwargs):
        k = _repeat_kv(key, module.num_key_value_groups)
        v = _repeat_kv(value, module.num_key_value_groups)
        sinks = module.sinks.reshape(1, -1, 1, 1)

        s_q = query.shape[-2]
        kv_len = k.shape[-2]
        leading = 1
        for d in query.shape[:-2]:
            leading *= d
        rows = max(1, min(s_q, budget_elems // max(1, leading * kv_len)))

        outs = []
        for s in range(0, s_q, rows):
            e = min(s + rows, s_q)
            qc = query[..., s:e, :]
            w = _torch.matmul(qc, k.transpose(2, 3)) * scaling
            if attention_mask is not None:
                m = attention_mask[..., s:e, :]
                w = w + m                              # fp32 mask promotes, like orig
            comb = _torch.cat([w, sinks.expand(query.shape[0], -1, w.shape[-2], -1)], dim=-1)
            comb = comb - comb.max(dim=-1, keepdim=True).values   # overflow clamp, like orig
            probs = _F.softmax(comb, dim=-1, dtype=comb.dtype)
            scores = probs[..., :-1]                   # drop the sink column
            scores = _F.dropout(scores, p=dropout, training=module.training)
            outs.append(_torch.matmul(scores.to(v.dtype), v))   # [B, H, rows, D]
        # cat chunks on the S axis, then orig's final transpose -> [B, S_q, H, D]
        out = _torch.cat(outs, dim=2).transpose(1, 2).contiguous()
        return out, None

    return chunked


def _patch_chunked_eager_attention():
    """Swap in the chunked eager attention for transformers' DeepSeek-V4 module.
    Idempotent; no-op if the module isn't imported yet."""
    import sys
    mod = sys.modules.get("transformers.models.deepseek_v4.modeling_deepseek_v4")
    if mod is None or getattr(mod, "_v4_chunk_patched", False):
        return
    mod.eager_attention_forward = _make_chunked_eager()
    mod._v4_chunk_patched = True


# =============================== telemetry ==============================

class TrainMonitor:
    """DeepSeek-style run analytics over the live loss/grad-norm streams.

    Keeps an EMA (fast-reacting) and a rolling median (robust baseline) of
    both series. A SPIKE is declared only when BOTH loss and grad-norm burst
    above k x their medians simultaneously (single-series jumps are routine),
    with a cooldown so one blowup doesn't spam alerts. On alert, `on_alert`
    receives a JSON-ready record — train() uses it to snapshot a pre-spike
    checkpoint you can roll back to."""

    def __init__(self, ema_beta: float = 0.98, median_window: int = 200,
                 loss_k: float = 1.5, gnorm_k: float = 3.0,
                 min_history: int = 10, cooldown: int = 50, on_alert=None):
        from collections import deque
        self.beta = ema_beta
        self.win = deque(maxlen=median_window)
        self.gwin = deque(maxlen=median_window)
        self.loss_ema = None
        self.gnorm_ema = None
        self.loss_k, self.gnorm_k = loss_k, gnorm_k
        self.min_history = min_history
        self.cooldown = cooldown
        self.last_alert_step = -(10 ** 9)
        self.n_alerts = 0
        self.on_alert = on_alert or (lambda rec: None)

    def update_train(self, step: int, loss: float, grad_norm: float, **extra):
        """Feed one train step; returns an alert record if this step spiked."""
        self.loss_ema = loss if self.loss_ema is None else \
            self.beta * self.loss_ema + (1 - self.beta) * loss
        self.gnorm_ema = grad_norm if self.gnorm_ema is None else \
            self.beta * self.gnorm_ema + (1 - self.beta) * grad_norm
        self.win.append(loss)
        self.gwin.append(grad_norm)

        if len(self.win) < max(self.min_history, 2):
            return None
        lmed, gmed = self.loss_median(), self.gnorm_median()
        if step - self.last_alert_step <= self.cooldown:
            return None
        if loss > self.loss_k * lmed and grad_norm > self.gnorm_k * gmed:
            self.last_alert_step = step
            self.n_alerts += 1
            rec = {"type": "alert", "step": step, "loss": round(loss, 5),
                   "loss_median": round(lmed, 5), "loss_ema": round(self.loss_ema, 5),
                   "grad_norm": round(grad_norm, 4), "gnorm_median": round(gmed, 4),
                   "gnorm_ema": round(self.gnorm_ema, 4), **extra}
            self.on_alert(rec)
            return rec
        return None

    def loss_median(self):
        import statistics
        return statistics.median(self.win)

    def gnorm_median(self):
        import statistics
        return statistics.median(self.gwin)


def render_curves(records, out_path: str, grok_lo: int = 0, grok_hi: int = 0):
    """3-panel PNG (loss / grad-norm / lr+throughput) from JSONL records.

    Written every CHART_EVERY steps + at run end, so a mid-run
    `modal volume get vision-adapter-data logs/train_curves.png` always
    shows the freshest curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = [r for r in records if r.get("type") == "train"]
    va = [r for r in records if r.get("type") == "val"]
    al = [r for r in records if r.get("type") == "alert"]
    if not tr:
        return False
    xs = [r["step"] for r in tr]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    ax.plot(xs, [r["loss"] for r in tr], lw=0.4, alpha=0.35, color="tab:blue")
    ema = _ema_series([r["loss"] for r in tr], 0.98)
    ax.plot(xs, ema, lw=1.8, color="tab:blue", label="train loss (EMA .98)")
    if va:
        ax.plot([r["step"] for r in va], [r["val_loss"] for r in va],
                "o-", ms=4, lw=1.2, color="tab:red", label="val loss")
    ax.set_yscale("log")
    ax.set_ylabel("loss (log)")
    ax.set_title("Vision-Adapter SFT — live curves")
    ax.legend(loc="upper right", fontsize=8)
    _shade_grok(ax, grok_lo, grok_hi)

    ax = axes[1]
    ax.plot(xs, [r["grad_norm"] for r in tr], lw=0.4, alpha=0.35, color="tab:green")
    ax.plot(xs, _ema_series([r["grad_norm"] for r in tr], 0.98),
            lw=1.8, color="tab:green", label="grad norm (EMA .98)")
    ax.set_yscale("log")
    ax.set_ylabel("||grad|| (log)")
    ax.legend(loc="upper right", fontsize=8)
    _shade_grok(ax, grok_lo, grok_hi)

    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(xs, [r["lr"] for r in tr], lw=1.2, color="tab:purple", label="lr")
    ax2.plot(xs, [r.get("tok_s", 0.0) for r in tr], lw=0.6, alpha=0.6,
             color="tab:orange", label="tokens/s")
    ax.set_ylabel("lr"); ax2.set_ylabel("tokens/s")
    ax.set_xlabel("step")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    for a in al:
        for ax_ in axes[:2]:
            ax_.axvline(a["step"], color="red", ls="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def _ema_series(vals, beta):
    out, e = [], None
    for v in vals:
        e = v if e is None else beta * e + (1 - beta) * v
        out.append(e)
    return out


def _shade_grok(ax, lo, hi):
    if hi > lo:
        ax.axvspan(lo, hi, color="gold", alpha=0.15)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], " grok window",
                fontsize=7, color="darkgoldenrod", va="top")


# =============================== train entry ==============

_T0 = time.time()


def _phase(msg):
    """Timed startup heartbeat (precompute contract): the backbone pull can
    take 15-30 min on a cold container — without these lines it looks hung."""
    print(f"[train] +{time.time() - _T0:6.1f}s  {msg}", flush=True)


def _shared_setup(offload: bool = True):
    import torch
    _phase("loading tokenizer ...")
    tok, model, proj = build_model(offload=offload)
    _phase(f"backbone+projector ready (trainable={sum(p.numel() for p in proj.parameters())/1e6:.1f}M)")
    ds = EmbSFT()
    collate = make_collate(tok, tok.pad_token_id)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=8, persistent_workers=True,
        pin_memory=True)

    # held-out val set (~2% of rows). Not shuffled; only used for the
    # periodic loss probe so "is it grokking?" isn't judged on train batches.
    val_ds = EmbSFT(manifest_rel=VAL_MANIFEST_REL)
    val_loader = None
    if len(val_ds):
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
            collate_fn=collate, num_workers=0)
    _phase(f"datasets ready (train={len(ds)} rows, val={len(val_ds)} rows)")

    import torch.optim as optim
    opt = optim.AdamW(proj.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    log_path = os.path.join(VOLUME_DIR, LOG_DIR_REL)
    os.makedirs(log_path, exist_ok=True)
    logger = open(os.path.join(log_path, LOG_FILE_REL), "a", buffering=1)
    return tok, model, proj, opt, loader, val_loader, logger


def _log(logger, rec):
    import json as _json
    logger.write(_json.dumps(rec) + "\n")


@torch.no_grad()
def _val_probe(model, proj, val_loader, tok):
    """Mean loss over the held-out split (projector-only, same recipe)."""
    import torch
    proj.eval()
    losses, n = [], 0
    for sig in val_loader:
        with visual_inject(sig, proj, model) as full:
            losses.append(float(model(**full).loss.item()))
        n += sig["input_ids"].shape[0]
    proj.train()
    return sum(losses) / max(1, len(losses)), n


def _dryrun_impl(mem_cap: float, offload: bool, compare_checkpointing: bool = False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    _props = torch.cuda.get_device_properties(0)
    print(f"[dryrun] gpu={_props.name} cc={_props.major}.{_props.minor} "
          f"vram={_props.total_memory / 2**30:.0f}GiB offload={'on' if offload else 'OFF (all-in-VRAM)'}",
          flush=True)
    tok, model, proj, opt, loader, _val_loader, _logger = _shared_setup(offload=offload)
    it = iter(loader)
    sig = next(it)
    step_out = _one_step(sig, model, proj, opt, tok)
    cur = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    # stable step-time baseline (excl. warmup) so Phase 5's B300 number has
    # something to compare against — this is the whole point of recording it.
    n_timed = min(4, max(1, len(loader) - 1))
    for _ in range(n_timed - 1):
        next(it)
    if hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_timed):
        _one_step(next(it), model, proj, opt, tok)
    if hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    step_s = (time.time() - t0) / n_timed
    line = (f"[dryrun] loss={step_out['loss']:.4f} n_trainable={sum(p.numel() for p in proj.parameters())/1e6:.1f}M "
            f"| mem_alloc={cur:.2f}GiB peak={peak:.2f}GiB budget={mem_cap:.0f}GiB -> "
            f"{'PASS' if peak < mem_cap else 'FAIL'} "
            f"| step={step_s:.2f}s ({1/step_s:.3f}it/s @ bs{int(BATCH_SIZE)}, n={n_timed})")
    print(line, flush=True)
    with open(os.path.join(VOLUME_DIR, "dryrun_report.txt"), "w") as f:
        f.write(line + "\n")
    vol.commit()

    if compare_checkpointing:
        # B300 has headroom: measure what turning grad-checkpointing OFF buys
        # (skips the 304B-forward recompute in backward — often -25..40%% step
        # time) and whether peak VRAM stays under the gate. Decision is made on
        # THESE numbers, not on estimates.
        model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(n_timed):
            _one_step(next(it), model, proj, opt, tok)
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
        step_off = (time.time() - t1) / n_timed
        peak_off = torch.cuda.max_memory_allocated() / 2**30
        line2 = (f"[dryrun] ckpt=OFF step={step_off:.2f}s ({1/step_off:.3f}it/s) "
                 f"peak={peak_off:.2f}GiB -> {'KEEP OFF' if peak_off < mem_cap else 'TOO HOT, keep ON'}")
        print(line2, flush=True)
        with open(os.path.join(VOLUME_DIR, "dryrun_report.txt"), "a") as f:
            f.write(line2 + "\n")
        model.gradient_checkpointing_enable()
    assert peak < mem_cap, "MEMORY GATE FAIL — do not run full train"


@app.function(image=train_image, gpu=GPU, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=3600, memory=f"{A100_CONTAINER_RAM_GB}GB")
def train_dryrun():
    """Phase 4 gate on the known-good stack: code correctness + A100 baseline."""
    _dryrun_impl(GPU_MEM_CAP_GIB, offload=True)


@app.function(image=train_image_b300, gpu=B300_GPU, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=3600, memory=f"{B300_CONTAINER_RAM_GB}GB")
def train_dryrun_b300():
    """Phase 5 gate: SM103/cu130 stack check, all-in-VRAM load, B300 step time
    vs the A100 baseline recorded by train_dryrun, plus a measured
    grad-checkpointing ON/OFF verdict."""
    _dryrun_impl(B300_GPU_MEM_CAP_GIB, offload=False, compare_checkpointing=True)


def _train_impl(offload: bool):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model, proj, opt, loader, val_loader, logger = _shared_setup(offload=offload)
    n_params = sum(p.numel() for p in proj.parameters())
    steps_total = EPOCHS * len(loader)
    grok_lo = int(SAMPLES_PER_BASETEN_GROK / BATCH_SIZE * 0.8)
    grok_hi = int(SAMPLES_PER_BASETEN_GROK / BATCH_SIZE * 1.25)
    print(f"[train] projector params={n_params/1e6:.2f}M | "
          f"grok window ≈ step {grok_lo}-{grok_hi} @ bs{int(BATCH_SIZE)} "
          f"(~{SAMPLES_PER_BASETEN_GROK} samples == Baseten's step-900 batch-64 recipe)")
    _log(logger, {"type": "run_start", "ts": round(time.time(), 1),
                  "config": {
                      "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                      "max_seq_len": MAX_SEQ_LEN, "steps_total": steps_total,
                      "n_trainable_params": n_params,
                      "backbone": DS_REPO, "projector_src": MOONVIT_REPO,
                      "train_manifest": DATASET_MANIFEST_REL,
                      "val_manifest": VAL_MANIFEST_REL,
                      "grad_clip": 1.0, "opt": "adamw(0.9,0.95)",
                      "gpu": GPU, "torch": torch.__version__,
                      "grok_window_steps": [grok_lo, grok_hi],
                  }})

    def _on_alert(rec):
        # snapshot the pre-spike weights so a blowup is recoverable without
        # waiting for the next periodic checkpoint.
        path = os.path.join(VOLUME_DIR, CKPT_DIR_REL, f"pre_spike_step{rec['step']}.pt")
        torch.save({"proj": proj.state_dict(), "step": rec["step"],
                    "loss": rec.get("loss"), "alert": rec}, path)
        vol.commit()
        print(f"[SPIKE-ALERT] step {rec['step']} loss={rec['loss']} "
              f"(med={rec['loss_median']}) gnorm={rec['grad_norm']} "
              f"(med={rec['gnorm_median']}) -> pre-spike ckpt saved: {os.path.basename(path)}",
              flush=True)

    monitor = TrainMonitor(on_alert=_on_alert)
    os.makedirs(os.path.join(VOLUME_DIR, CKPT_DIR_REL), exist_ok=True)
    step = 0
    t0 = time.time()
    samples_seen = 0
    tokens_seen = 0
    records = []
    for epoch in range(EPOCHS):
        for sig in loader:
            step += 1
            out = _one_step(sig, model, proj, opt, tok)
            samples_seen += int(BATCH_SIZE)
            tokens_seen += out["tokens"]
            alert = monitor.update_train(step=step, loss=out["loss"],
                                         grad_norm=out["grad_norm"], lr=LR)
            if alert:
                _log(logger, alert)
                records.append(alert)
            elapsed = time.time() - t0
            rec = {
                "step": step, "epoch": epoch, "type": "train",
                "loss": round(out["loss"], 5),
                "loss_ema": round(monitor.loss_ema, 5),
                "grad_norm": round(out["grad_norm"], 4),
                "gnorm_ema": round(monitor.gnorm_ema, 4),
                "samples_seen": samples_seen,
                "tokens_seen": tokens_seen,
                "tok_s": round(tokens_seen / max(1e-9, elapsed), 1),
                "lr": float(opt.param_groups[0]["lr"]),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "it_s": round(step / max(1e-9, elapsed), 3),
                "step_ms": out["step_ms"],
                "eta_min": round((steps_total - step) * out["step_ms"] / 1000 / 60, 1),
                "ts": round(time.time(), 1),
            }
            records.append(rec)
            if step % LOG_EVERY == 0:
                _log(logger, rec)
            if step % 20 == 0:
                rate = step / max(1e-9, elapsed)
                print(f"[train] e{epoch} step {step}/{steps_total} "
                      f"loss={out['loss']:.4f} ema={monitor.loss_ema:.4f} "
                      f"gnorm={out['grad_norm']:.2f} tok/s={rec['tok_s']:.0f} "
                      f"samples={samples_seen} peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB "
                      f"{rate:.2f}it/s ETA={rec['eta_min']:.0f}m"
                      + (" ALERTS=%d" % monitor.n_alerts if monitor.n_alerts else ""),
                      flush=True)
                torch.save({"proj": proj.state_dict(), "step": step, "loss": float(out["loss"])},
                           os.path.join(VOLUME_DIR, CKPT_DIR_REL, f"latest.pt"))
                vol.commit()
            if val_loader and step % VAL_EVERY == 0:
                # quick probe on the held-out split, thenPROJECTOR back to train mode
                vloss, n = _val_probe(model, proj, val_loader, tok)
                vrec = {"step": step, "epoch": epoch, "type": "val",
                        "val_loss": round(vloss, 5), "n_rows": n,
                        "samples_seen": samples_seen,
                        "ts": round(time.time(), 1)}
                _log(logger, vrec)
                records.append(vrec)
                print(f"[val]   e{epoch} step {step} val_loss={vloss:.4f} (n={n})", flush=True)
            if step % VAL_EVERY == 0 or step == steps_total:
                # curves PNG on the volume — fetch mid-run with:
                #   modal volume get vision-adapter-data logs/train_curves.png
                try:
                    render_curves(records, os.path.join(VOLUME_DIR, LOG_DIR_REL, "train_curves.png"),
                                  grok_lo=grok_lo, grok_hi=grok_hi)
                    vol.commit()
                except Exception as e:   # charting must never kill training
                    print(f"[train] chart render failed (ignored): {e}", flush=True)
            if step % SAVE_EVERY == 0:
                torch.save(proj.state_dict(),
                           os.path.join(VOLUME_DIR, CKPT_DIR_REL, f"projector_step{step}.safetensors"))
                vol.commit()
            if GROK_STOP_AFTER_STEPS and step >= GROK_STOP_AFTER_STEPS:
                print(f"[train] early stop: hit GROK_STOP_AFTER_STEPS={GROK_STOP_AFTER_STEPS}")
                break
    torch.save(proj.state_dict(), os.path.join(VOLUME_DIR, CKPT_DIR_REL, "projector_final.safetensors"))
    render_curves(records, os.path.join(VOLUME_DIR, LOG_DIR_REL, "train_curves.png"),
                  grok_lo=grok_lo, grok_hi=grok_hi)
    _log(logger, {"type": "run_end", "step": step, "samples_seen": samples_seen,
                  "tokens_seen": tokens_seen, "n_alerts": monitor.n_alerts,
                  "wall_min": round((time.time() - t0) / 60, 1), "ts": round(time.time(), 1)})
    vol.commit()
    logger.close()
    print(f"[train] DONE after {step} steps | alerts={monitor.n_alerts} "
          f"| wall={(time.time() - t0)/60:.0f}min")


@app.function(image=train_image, gpu=GPU, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=86400, memory=f"{A100_CONTAINER_RAM_GB}GB")
def train():
    """Phase 4/5 fallback trainer on the known-good A100 stack."""
    _train_impl(offload=True)


@app.function(image=train_image_b300, gpu=B300_GPU, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=86400, memory=f"{B300_CONTAINER_RAM_GB}GB")
def train_b300():
    """Phase 5 target: full SFT all-in-VRAM on B300 (run only after train_dryrun_b300 PASS)."""
    _train_impl(offload=False)


def _one_step(sig, model, proj, opt, tok):
    import torch
    import torch.nn as nn
    t0 = time.time()
    with visual_inject(sig, proj, model) as full:
        out = model(**full)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip at 1.0 AND capture the PRE-clip total norm: grad-norm bursts are the
        # earliest warning of a loss cliff (they move before the loss does).
        gnorm = nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()
    n_tokens = int(sig["attention_mask"].sum())
    return {"loss": float(loss.item()),
            "grad_norm": float(gnorm),
            "tokens": n_tokens,
            "step_ms": round((time.time() - t0) * 1000, 1)}
