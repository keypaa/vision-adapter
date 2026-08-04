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
  - the n_vis vision embeddings are INSERTED via inputs_embeds at positions
    [1 : 1+n_vis]; attention_mask covers them with 1s;
  - labels: -100 everywhere EXCEPT the answer + EOS token positions.

Stages:
  train_dryrun : ONE fwd/bwd at batch=8; asserts peak VRAM < 70 GiB; else ABORT.
  train        : full SFT loop (LR 5e-4, bs 8, adamw) once dry run passes.

Run:    modal run modal_train.py::train_dryrun     # then
        modal run modal_train.py::train
"""
from __future__ import annotations
import os, json, glob, time
import modal

GPU = "A100-80GB"
GPU_MEM_CAP_GIB = 70.0
SYS_RAM_CAP_GIB = 200
BATCH_SIZE = 8
LR = 5e-4
MAX_SEQ_LEN = 4096
EPOCHS = 2

VOL_NAME = "vision-adapter-data"
EMB_ROOT_REL = "embeddings"
CKPT_DIR_REL = "checkpoints"
DATASET_MANIFEST_REL = "train_manifest.jsonl"  # produced by Phase3 etl+mix stage

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
    )
    .env({"HF_HOME": HF_CACHE})
)

app = modal.App("vision-adapter-train")


# ============================ model assembly (shared) =========================

def build_model():
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DS_REPO, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        DS_REPO,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: f"{int(GPU_MEM_CAP_GIB)}GiB", "cpu": f"{int(SYS_RAM_CAP_GIB)}GiB"},
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

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
    """

    def __init__(self, vol_dir=VOLUME_DIR):
        self.vol = vol_dir
        self.rows = []
        with open(os.path.join(vol_dir, DATASET_MANIFEST_REL)) as f:
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
            a = a[: max(1, budget_text - len(u))]      # keep answer; trim user if overflowing
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


def inject_visual(inputs, proj, model):
    """Project cached ViT embeddings -> LLM dim, embed tokens, and splice the
    visual block at positions [1 : 1+n_vis], producing inputs_embeds for the LLM."""
    import torch
    with torch.no_grad():
        ids = inputs["input_ids"].clamp_min(0).to("cuda")
        text_emb = model.get_input_embeddings()(ids)      # [B,L,4096]
    vis = proj(inputs["vis"].to("cuda", torch.bfloat16))  # [B,maxv,4096]
    merged = text_emb.clone()
    attn = inputs["attention_mask"].clone()
    labels = inputs["labels"].clone()
    n_vis = inputs["n_vis"].tolist()
    for i in range(ids.shape[0]):
        nv = int(n_vis[i])
        merged[i, 1: 1 + nv] = vis[i, :nv]
        # positions [1:1+nv] are image: exclude from labels (defensive; already -100)
        labels[i, 1: 1 + nv] = -100
    return {
        "inputs_embeds": merged,
        "attention_mask": attn.to("cuda"),
        "labels": labels.to("cuda"),
    }


# =============================== train entry ==============

def _shared_setup():
    import torch
    tok, model, proj = build_model()
    ds = EmbSFT()
    collate = make_collate(tok, tok.pad_token_id)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=2, persistent_workers=True)
    import torch.optim as optim
    opt = optim.AdamW(proj.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    return tok, model, proj, opt, loader


@app.function(image=train_image, gpu=GPU, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=3600, memory=f"{SYS_RAM_CAP_GIB}GB")
def train_dryrun():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model, proj, opt, loader = _shared_setup()
    sig = next(iter(loader))
    step_out = _one_step(sig, model, proj, opt, tok)
    cur = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    line = (f"[dryrun] loss={step_out['loss']:.4f} n_trainable={sum(p.numel() for p in proj.parameters())/1e6:.1f}M "
            f"| mem_alloc={cur:.2f}GiB peak={peak:.2f}GiB budget={GPU_MEM_CAP_GIB:.0f}GiB -> "
            f"{'PASS' if peak < GPU_MEM_CAP_GIB else 'FAIL'}")
    print(line)
    with open(os.path.join(VOLUME_DIR, "dryrun_report.txt"), "w") as f:
        f.write(line + "\n")
    vol.commit()
    assert peak < GPU_MEM_CAP_GIB, "MEMORY GATE FAIL — do not run full train"


@app.function(image=train_image, gpu=GPU, volumes={VOLUME_DIR: vol, HF_CACHE: hf_vol},
              timeout=86400, memory=f"{SYS_RAM_CAP_GIB}GB")
def train():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model, proj, opt, loader = _shared_setup()
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"[train] projector params={n_params/1e6:.2f}M | target grok ≈ step 900 @ 64-eff batch")
    os.makedirs(os.path.join(VOLUME_DIR, CKPT_DIR_REL), exist_ok=True)
    step = 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        for sig in loader:
            step += 1
            out = _one_step(sig, model, proj, opt, tok)
            if step % 20 == 0:
                rate = step / max(1e-9, time.time() - t0)
                print(f"[train] e{epoch} step {step} loss={out['loss']:.4f} "
                      f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB {rate:.2f}it/s")
                torch.save({"proj": proj.state_dict(), "step": step, "loss": float(out["loss"])},
                           os.path.join(VOLUME_DIR, CKPT_DIR_REL, f"latest.pt"))
                vol.commit()
            if step % 200 == 0:
                torch.save(proj.state_dict(),
                           os.path.join(VOLUME_DIR, CKPT_DIR_REL, f"projector_step{step}.safetensors"))
                vol.commit()
    torch.save(proj.state_dict(), os.path.join(VOLUME_DIR, CKPT_DIR_REL, "projector_final.safetensors"))
    vol.commit()
    print(f"[train] DONE after {step} steps")


def _one_step(sig, model, proj, opt, tok):
    import torch
    import torch.nn as nn
    full = inject_visual(sig, proj, model)
    out = model(**full)
    loss = out.loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
    opt.step()
    return {"loss": float(loss.item())}
