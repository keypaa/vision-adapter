"""vision_adapter/core.py — single shared training core.

Replaces the ~40% duplication across:
  grok_probe_qwen.py  (HourglassProjector / make_collate+check / embeds_for / lr_at / ProbeMonitor / render_curves)
  modal_probe.py      (same + make_collate bos guard port)
  modal_train.py      (HourglassProjector inside build_model / make_collate sans guard / visual_inject / TrainMonitor / render_curves)

This file is the 80% value with 5% churn — the LayerNorm dtype bug had to be
fixed twice because the same F32→bf16 cast lived in two places. All three
trainers import from here and re-export aliases so existing tests
(`test_train_collate` imports from modal_train, `test_grok_probe_smoke`
imports from grok_probe_qwen) keep passing.

Keep this file dependency-light: only torch + stdlib. No modal, no HF.
"""
from __future__ import annotations

import math
import statistics
import time
from collections import deque

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# HourglassProjector — LN(vision) -> Linear(vision, 2*llm) -> GELU -> Linear(2*llm, llm)
# ---------------------------------------------------------------------------

class HourglassProjector(nn.Module):
    """Same structure everywhere, dims adapted to the chosen LLM hidden size.

    For DeepSeek d==h==4096 so up/down symmetric (4096->8192->4096); for Qwen
    2B/4B hidden 2048/2560 the hourglass is 4096->4096/5120->2048/2560.
    Reads vision_dim/llm_dim at construction, not from a global.
    """

    def __init__(self, vision_dim: int = 4096, llm_dim: int = 2048):
        super().__init__()
        self.ln = nn.LayerNorm(vision_dim)
        self.up = nn.Linear(vision_dim, 2 * llm_dim)
        self.act = nn.GELU()
        self.dn = nn.Linear(2 * llm_dim, llm_dim)

    def forward(self, x):
        return self.dn(self.act(self.up(self.ln(x))))


# ---------------------------------------------------------------------------
# make_collate — [BOS?][img × n_vis][user][answer][EOS] with answer priority
# ---------------------------------------------------------------------------

def make_collate(tok, pad_id: int, max_len: int = 4096, vision_dim: int = 4096):
    """Identical to modal_train.make_collate but with the Qwen bos=None guard.

    Layout per example: [BOS?][img × n_vis][user][answer][EOS]
    - answer has PRIORITY on the text budget (it is the loss target)
    - labels -100 everywhere except answer+EOS; BOS/img/user/pad never contribute
    - attention_mask covers BOS..EOS (incl. img span), not right-pad
    - vis tokens padded to batch max with zeros, dtype float32

    The bos guard (`if tok.bos_token_id is not None`) is the fix ported from
    grok_probe/modal_probe — DeepSeek has BOS, Qwen3.5 has None. Without it
    Qwen raises on `LongTensor assignment of None`.
    """

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
        vis_pad = torch.zeros(B, max_v, vision_dim, dtype=torch.float32)
        for i, b in enumerate(batch):
            vis_pad[i, : n_vis[i]] = b["vis"]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels,
                "vis": vis_pad, "n_vis": torch.tensor(n_vis, dtype=torch.long),
                "g": [b.get("g", "?") for b in batch]}

    return collate


def check_collate_invariants(batch, tok, vision_dim: int = 4096) -> None:
    """First-batch port of test_train_collate.py's pins — abort loudly rather
    than train on a drifted sequence contract."""
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    nv = int(batch["n_vis"][0])
    assert ids[0] == tok.bos_token_id, "position 0 must be BOS"
    sup = [i for i, lab in enumerate(labels) if lab != -100]
    assert sup and sup[-1] == ids.index(tok.eos_token_id), "last supervised position must be EOS"
    assert labels[sup[-1]] == tok.eos_token_id, "EOS must carry its own label"
    assert all(labels[j] == -100 for j in range(sup[0])), \
        "nothing before the answer span may carry labels (BOS/img/user excluded)"
    assert all(i > 1 + nv for i in sup), "supervised positions must lie past the image span"
    assert sup == list(range(sup[0], sup[-1] + 1)), "answer span must be contiguous"
    assert batch["vis"].shape[-1] == vision_dim and batch["vis"].dtype == torch.float32


# ---------------------------------------------------------------------------
# embeds_for — Qwen path: inputs_embeds-only injection at [1:1+n_vis]
# ---------------------------------------------------------------------------

def embeds_for(model, batch, proj, device):
    """inputs_embeds-only injection (Qwen): base rows from frozen embedding
    table, visual span overwritten by projected embeddings (autograd flows
    through those rows to the projector). Contrast with visual_inject below
    which uses a hook so input_ids stays the model input (DeepSeek hash-MoE)."""
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out_dtype = next(model.parameters()).dtype
    proj_dtype = next(proj.parameters()).dtype
    with torch.no_grad():
        base = model.get_input_embeddings()(ids).to(out_dtype)
    vis = batch["vis"].to(device).to(proj_dtype)
    pv = proj(vis).to(out_dtype)
    merged = base.clone()
    for i, nv in enumerate(batch["n_vis"].tolist()):
        merged[i, 1: 1 + nv] = pv[i, :nv]
    return {"inputs_embeds": merged, "attention_mask": attn, "labels": labels}


# ---------------------------------------------------------------------------
# visual_inject — DeepSeek path: hook embed_tokens, splice [1:1+n_vis]
# ---------------------------------------------------------------------------

class visual_inject:
    """Project cached ViT embeddings -> LLM dim and splice at [1:1+n_vis].

    DeepSeek-V4's hash-MoE gates pick experts via tid2eid[input_ids] and the
    core model raises if input_ids AND inputs_embeds are both given, so plain
    inputs_embeds injection cannot work. We keep input_ids as the model input
    and hook embed_tokens, overwriting its OUTPUT rows at the visual positions.
    Gradients reach the projector through the spliced rows.

        with visual_inject(sig, proj, model) as full:
            out = model(**full)
    """

    def __init__(self, inputs, proj, model):
        embed = model.get_input_embeddings()
        self._embed = embed
        self._dev = next(embed.parameters()).device
        self._vis = proj(inputs["vis"].to(self._dev, torch.bfloat16))
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
            merged = output.clone()
            for i, nv in enumerate(n_vis):
                merged[i, 1: 1 + nv] = vis[i, :nv]
            return merged

        self._handle = self._embed.register_forward_hook(_splice)
        return self.model_inputs

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


# ---------------------------------------------------------------------------
# lr_at — linear warmup then cosine decay to 10%
# ---------------------------------------------------------------------------

def lr_at(step: int, max_steps: int, base_lr: float = 5e-4, warmup: int = 100) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step <= warmup:
        return base_lr * max(1, step) / warmup
    t = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


# ---------------------------------------------------------------------------
# ProbeMonitor — EMA + plateau banner + spike + collapse (Qwen probe)
# ---------------------------------------------------------------------------

class ProbeMonitor:
    """EMA loss stream + flat-loss plateau banners + >2x EMA-jump alerts."""

    def __init__(self, plateau_window: int = 300, plateau_check_every: int = 50,
                 plateau_rel_tol: float = 0.02, ema_beta: float = 0.98,
                 spike_factor: float = 2.0, spike_window: int = 100,
                 spike_min_history: int = 20, samples_per_baseten_grok: int = 57600):
        self.plateau_window = plateau_window
        self.plateau_check_every = plateau_check_every
        self.plateau_rel_tol = plateau_rel_tol
        self.ema_beta = ema_beta
        self.spike_factor = spike_factor
        self.spike_window = spike_window
        self.spike_min_history = spike_min_history
        self.samples_per_baseten_grok = samples_per_baseten_grok
        self.ema = None
        self.prev_ema = None
        self.history: deque = deque(maxlen=plateau_window)
        self.ema_history: deque = deque(maxlen=spike_window)
        self.last_banner_step = -(10 ** 9)
        self.last_alert_step = -(10 ** 9)
        self.n_alerts = 0
        self.n_banners = 0
        self.collapse_step = None

    def update(self, step: int, loss: float, samples_seen: int):
        self.prev_ema = self.ema
        self.ema = loss if self.ema is None else self.ema_beta * self.ema + (1 - self.ema_beta) * loss
        self.history.append(loss)
        self.ema_history.append(self.ema)

        if (len(self.ema_history) >= self.spike_min_history
                and step - self.last_alert_step > 50
                and self.ema > self.spike_factor * min(self.ema_history)):
            print(f"[SPIKE-ALERT] step {step}: ema_loss {self.ema:.4f} > "
                  f"{self.spike_factor:.0f}x recent floor "
                  f"{min(self.ema_history):.4f}", flush=True)
            self.last_alert_step = step
            self.n_alerts += 1

        if self.collapse_step is None and len(self.history) >= self.plateau_window:
            base = statistics.median(self.history)
            recent = sum(list(self.history)[-self.plateau_check_every:]) / self.plateau_check_every
            if recent < 0.5 * base:
                self.collapse_step = step
                print(f"\n*** COLLAPSE *** step={step} samples_seen={samples_seen} "
                      f"(recent-mean {recent:.4f} < 0.5 * window-median {base:.4f})\n",
                      flush=True)

        if (self.prev_ema is not None and step - self.last_alert_step > 50
                and self.ema > self.spike_factor * self.prev_ema):
            print(f"[SPIKE-ALERT] step {step}: ema_loss {self.prev_ema:.4f} -> "
                  f"{self.ema:.4f} (> {self.spike_factor:.0f}x jump)", flush=True)
            self.last_alert_step = step
            self.n_alerts += 1

        if (step - self.last_banner_step >= self.plateau_check_every
                and len(self.history) >= self.plateau_window):
            old = self.history[0]
            if old > 0 and abs(self.ema - old) / old < self.plateau_rel_tol:
                print(
                    "\n" + "=" * 74 + "\n"
                    "[plateau] FLAT LOSS IS EXPECTED DURING THE GROK PHASE.\n"
                    f"[plateau] reference: collapse at ~{self.samples_per_baseten_grok} samples_seen\n"
                    "[plateau] (Baseten GLM-5.2 recipe: batch 64 / step ~900 / 66k imgs).\n"
                    f"[plateau] current samples_seen={samples_seen} "
                    f"({100 * samples_seen / self.samples_per_baseten_grok:.1f}% of reference)\n"
                    "[plateau] keep going — do not restart because the curve looks dead.\n"
                    + "=" * 74 + "\n",
                    flush=True)
                self.last_banner_step = step
                self.n_banners += 1
        return self.ema


# ---------------------------------------------------------------------------
# TrainMonitor — DeepSeek run analytics (loss+grad median spike detector)
# ---------------------------------------------------------------------------

class TrainMonitor:
    """DeepSeek-style run analytics over loss/grad-norm streams.

    Keeps EMA and rolling median of both series. SPIKE only when BOTH burst
    above k x median simultaneously, with cooldown so one blowup doesn't spam.
    """

    def __init__(self, ema_beta: float = 0.98, median_window: int = 200,
                 loss_k: float = 1.5, gnorm_k: float = 3.0,
                 min_history: int = 10, cooldown: int = 50, on_alert=None):
        self.beta = ema_beta
        self.win: deque = deque(maxlen=median_window)
        self.gwin: deque = deque(maxlen=median_window)
        self.loss_ema = None
        self.gnorm_ema = None
        self.loss_k, self.gnorm_k = loss_k, gnorm_k
        self.min_history = min_history
        self.cooldown = cooldown
        self.last_alert_step = -(10 ** 9)
        self.n_alerts = 0
        self.on_alert = on_alert or (lambda rec: None)

    def update_train(self, step: int, loss: float, grad_norm: float, **extra):
        """Feed one train step; returns alert record if spiked."""
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
        return statistics.median(self.win)

    def gnorm_median(self):
        return statistics.median(self.gwin)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _ema_series(vals, beta: float = 0.98):
    out, e = [], None
    for v in vals:
        e = v if e is None else beta * e + (1 - beta) * v
        out.append(e)
    return out


def _shade_grok(ax, lo: int, hi: int) -> None:
    if hi > lo:
        ax.axvspan(lo, hi, color="gold", alpha=0.15)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], " grok window",
                fontsize=7, color="darkgoldenrod", va="top")


# ---------------------------------------------------------------------------
# render_curves — 2-panel probe (grok) and 3-panel train (DeepSeek)
# ---------------------------------------------------------------------------

def render_curves(records: list[dict], out_path: str) -> bool:
    """2-panel probe PNG: raw loss + EMA + LR (log-y), grad-norm below;
    secondary x-axis in samples_seen on BOTH panels (telemetry contract)."""
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[probe] chart import failed (ignored): {e}", flush=True)
        return False
    if not records:
        return False

    # probe records store gnorm; normalize to train form for shared code
    tr = [r for r in records if r.get("type") in ("train", None) or "loss" in r]
    # filter to those with step (exclude config_header)
    tr = [r for r in tr if "step" in r]
    if not tr:
        return False
    xs = [r["step"] for r in tr]
    ss = [r.get("samples_seen", r["step"] * 8) for r in tr]

    def secax(ax):
        s2 = ax.secondary_xaxis("top", functions=(
            lambda s: s * ss[-1] / max(1, xs[-1]),
            lambda v: v * max(1, xs[-1]) / max(1, ss[-1])))
        s2.set_xlabel("samples_seen")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    ax = axes[0]
    ax.plot(xs, [r["loss"] for r in tr], lw=0.4, alpha=0.35, color="tab:blue", label="train loss (raw)")
    # ema key is ema_loss (probe) or loss_ema (train); handle both
    ema_vals = [r.get("ema_loss", r.get("loss_ema", r["loss"])) for r in tr]
    ax.plot(xs, ema_vals, lw=1.8, color="tab:blue", label="loss EMA(.98)")
    if any("lr" in r for r in tr):
        ax.plot(xs, [r.get("lr", 0) for r in tr], lw=1.0, color="tab:purple", label="lr")
    ax.set_yscale("log")
    ax.set_ylabel("loss (log) / lr")
    ax.set_xlabel("steps")
    ax.set_title(f"grok-probe live curves — {len(tr)} steps")
    ax.legend(loc="upper right", fontsize=8)
    secax(ax)

    ax = axes[1]
    gvals = [r.get("gnorm", r.get("grad_norm", 0)) for r in tr]
    ax.plot(xs, gvals, lw=0.5, alpha=0.5, color="tab:green", label="grad norm (pre-clip)")
    ax.set_yscale("log")
    ax.set_ylabel("||grad|| (log)")
    ax.set_xlabel("steps")
    ax.legend(loc="upper right", fontsize=8)
    secax(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def render_train_curves(records, out_path: str, grok_lo: int = 0, grok_hi: int = 0) -> bool:
    """3-panel PNG (loss / grad-norm / lr+throughput) from JSONL records."""
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[train] chart import failed (ignored): {e}", flush=True)
        return False

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
    ax.set_ylabel("lr")
    ax2.set_ylabel("tokens/s")
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


def train_step_qwen(model, proj, opt, batch, device, clip: float = 1.0, scaler=None) -> dict:
    """One fwd/bwd/clip/step for the Qwen probe (selective lm_head loss).

    Shared between grok_probe_qwen and modal_probe; modal_train keeps its own
    _one_step that goes through visual_inject (hash-MoE hook).
    """
    import torch.nn.functional as F
    t0 = time.time()
    amp_dtype = None
    if device == "cuda" and next(model.parameters()).dtype == torch.float32:
        amp_dtype = torch.float16
    inp = embeds_for(model, batch, proj, device)
    labels = inp.pop("labels")
    base = model.model
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        out = base(inputs_embeds=inp["inputs_embeds"], attention_mask=inp["attention_mask"])
        hidden = out.last_hidden_state
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        pos = mask.nonzero(as_tuple=False)
        h_sel = hidden[:, :-1][pos[:, 0], pos[:, 1]]
        y_sel = shift_labels[pos[:, 0], pos[:, 1]]
        logits_sel = model.lm_head(h_sel).float()
    loss = F.cross_entropy(logits_sel, y_sel)

    params = list(proj.parameters())
    if scaler is not None:
        scaled_loss = scaler.scale(loss)
        opt.zero_grad(set_to_none=True)
        scaled_loss.backward()
        scaler.unscale_(opt)
        gnorm = float(nn.utils.clip_grad_norm_(params, clip))
        step_skipped = math.isnan(gnorm) or math.isinf(gnorm)
        if not step_skipped:
            scaler.step(opt)
        scaler.update()
        finite = not step_skipped
    else:
        finite = bool(torch.isfinite(loss))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float("nan") if not finite else float(nn.utils.clip_grad_norm_(params, clip))
        if finite:
            opt.step()
    return {"loss": float(loss.item()), "finite": finite, "gnorm": gnorm,
            "tokens": int(batch["attention_mask"].sum()),
            "batch_size": int(batch["input_ids"].shape[0]),
            "step_ms": round((time.time() - t0) * 1000, 1)}
