"""Standalone MoonViT-V2 (moonvit3d) vision encoder, extracted from moonshotai/Kimi-K3.

Reproduces the encoder exactly so the 401M BF16 weights in
keypa/MoonViT-V2-Standalone can be run without the 2.8T checkpoint.

Contract (verified against modeling_kimi_k3.py):
  forward(pixel_values, grid_thws)
    pixel_values : (total_patches, 3, 14, 14) float  -- pre-patchified, packed
    grid_thws    : (num_images, 3) int64             -- rows [t, h, w] in patch units
  returns        : list[(n_merged_i, 4, 1024)]       -- one tensor per image, after 2x2 sd2_tpool

Weights are stored in the standalone safetensors under the canonical Kimi
`vision_tower.` prefix; load_moonvit_from_safetensors() remaps them onto `_vt.*`:
  vision_tower.patch_embed.proj.weight                    [1024, 3, 14, 14]
  vision_tower.patch_embed.pos_emb.weight                 [64, 64, 1024]   (learnable, bilinear interp)
  vision_tower.encoder.blocks.{i}.norm0.weight            [1024]
  vision_tower.encoder.blocks.{i}.wqkv.weight             [4608, 1024]
  vision_tower.encoder.blocks.{i}.wo.weight               [1536, 1024]
  vision_tower.encoder.blocks.{i}.norm1.weight            [1024]
  vision_tower.encoder.blocks.{i}.mlp.fc0.weight          [4096, 1024]
  vision_tower.encoder.blocks.{i}.mlp.fc1.weight          [1024, 4096]
  vision_tower.encoder.final_layernorm.weight             [1024]
"""
from __future__ import annotations
import json, math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 14


# ----------------------------- positional embeddings -----------------------------

def _bilinear_resize_grid(weight: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """weight: [64,64,1024] -> [h*w,1024] via bilinear interp (divided_fixed)."""
    gh, gw, d = weight.shape
    if (gh, gw) == (h, w):
        return weight.reshape(h * w, d)
    x = weight.permute(2, 0, 1).unsqueeze(0).float()          # [1,1024,64,64]
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.squeeze(0).permute(1, 2, 0).reshape(h * w, d)     # [h*w,1024]


class Rope2D(nn.Module):
    """2D rotary position embedding (Rope2DPosEmbRepeated), theta=10000."""
    def __init__(self, dim: int, max_h: int = 512, max_w: int = 512, theta: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        inv = 1.0 / (theta ** (torch.arange(0, dim, 4, dtype=torch.float32) / dim))
        t_h = torch.arange(max_h, dtype=torch.float32)
        t_w = torch.arange(max_w, dtype=torch.float32)
        # split-half frequency basis: cos[i], sin[i] pair with (x_i, x_{i+dim/2})
        ang_h = torch.outer(t_h, inv)         # [max_h, dim/4]
        ang_w = torch.outer(t_w, inv)         # [max_w, dim/4]
        self.register_buffer("cos_h", torch.cos(ang_h), persistent=False)
        self.register_buffer("sin_h", torch.sin(ang_h), persistent=False)
        self.register_buffer("cos_w", torch.cos(ang_w), persistent=False)
        self.register_buffer("sin_w", torch.sin(ang_w), persistent=False)

    def get(self, h: int, w: int):
        """Return cos/sin of shape [h*w, dim/2]: first dim/4 from h-axis, last dim/4 from w-axis."""
        ch = self.cos_h[:h].unsqueeze(1).expand(h, w, -1).reshape(h * w, -1)
        sh = self.sin_h[:h].unsqueeze(1).expand(h, w, -1).reshape(h * w, -1)
        cw = self.cos_w[:w].unsqueeze(0).expand(h, w, -1).reshape(h * w, -1)
        sw = self.sin_w[:w].unsqueeze(0).expand(h, w, -1).reshape(h * w, -1)
        return torch.cat([ch, cw], dim=-1).float(), torch.cat([sh, sw], dim=-1).float()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [seq, heads, dim]. Split-half rotary: pair (x_i, x_{i+d/2}) rotates by theta_i.
    cos/sin: [seq, dim/2]."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    c = cos.unsqueeze(1); s = sin.unsqueeze(1)
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.cat([out1, out2], dim=-1).type_as(x)


# ----------------------------- encoder -----------------------------

class _Mlp(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.fc0 = nn.Linear(hidden, inter, bias=False)
        self.fc1 = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class EncoderLayer(nn.Module):
    def __init__(self, hidden=1024, qkv_hidden=1536, inter=4096, heads=12, eps=1e-5):
        super().__init__()
        self.heads = heads
        self.head_dim = qkv_hidden // heads            # 128
        self.qkv_hidden = qkv_hidden
        self.norm0 = nn.RMSNorm(hidden, eps=eps)
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.wqkv = nn.Linear(hidden, 3 * qkv_hidden, bias=False)
        self.wo = nn.Linear(qkv_hidden, hidden, bias=False)
        self.mlp = _Mlp(hidden, inter)                 # -> blocks.i.mlp.fc0/fc1

    def forward(self, x, rope_cs, cu_seqlens, max_seqlen):
        b_total = x.shape[0]
        h = self.norm0(x)
        qkv = self.wqkv(h)
        q, k, v = qkv.split(self.qkv_hidden, dim=-1)
        q = q.reshape(b_total, self.heads, self.head_dim)
        k = k.reshape(b_total, self.heads, self.head_dim)
        v = v.reshape(b_total, self.heads, self.head_dim)
        _cos, _sin = rope_cs
        q = _apply_rope(q, _cos, _sin); k = _apply_rope(k, _cos, _sin)

        attn_out = None
        try:
            from flash_attn import flash_attn_varlen_func
            attn_out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                                              max_seqlen, max_seqlen, causal=False)
        except Exception:
            outs, prev = [], 0
            cu = cu_seqlens.tolist() + [b_total]
            for i in range(len(cu)):
                s = cu[i]; e = cu[i + 1] if i + 1 < len(cu) else b_total
                if e <= s: continue
                qi = q[s:e].transpose(0, 1).unsqueeze(0)  # [1,heads,seq,dim]
                ki = k[s:e].transpose(0, 1).unsqueeze(0)
                vi = v[s:e].transpose(0, 1).unsqueeze(0)
                oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)
                outs.append(oi.squeeze(0).transpose(0, 1))
            attn_out = torch.cat(outs, dim=0) if outs else q

        x = x + self.wo(attn_out.reshape(b_total, self.qkv_hidden))
        x = x + self.mlp(self.norm1(x))
        return x


class PosEmb(nn.Module):
    """Learnable 2D divided_fixed positional embedding [64,64,1024] (loadable)."""
    def __init__(self, hidden=1024, gh=64, gw=64):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(gh, gw, hidden))
        self.gh, self.gw = gh, gw


class PatchEmbed(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.proj = nn.Conv2d(3, hidden, kernel_size=PATCH, stride=PATCH, bias=False)
        self.pos_emb = PosEmb(hidden)   # -> patch_embed.pos_emb.weight

    def forward(self, patches, grid_hw):  # patches: [total,3,14,14]; grid_hw: list[(h,w)] in patch units
        feats = self.proj(patches)                    # [total,1024,1,1]
        feats = feats.flatten(2).squeeze(-1)          # [total,1024]
        pos_parts = []
        for (h, w) in grid_hw:
            pos = _bilinear_resize_grid(self.pos_emb.weight, h, w)   # [h*w,1024]
            pos_parts.append(pos)
        pos = torch.cat(pos_parts, dim=0).to(feats.dtype) if pos_parts else 0
        return feats + pos


class _Encoder(nn.Module):
    """Named 'encoder' so its children bind to checkpoint keys encoder.blocks.* / encoder.final_layernorm.*"""
    def __init__(self, hidden, qkvmain, inter, layers, heads):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderLayer(hidden, qkvmain, inter, heads) for _ in range(layers)])
        self.final_layernorm = nn.RMSNorm(hidden, eps=1e-5)


class MoonViTV2(nn.Module):
    """Frozen standalone MoonViT-V2; returns per-image merged [n,4,1024] lists.

    Loader binds canonical Kimi key prefix `vision_tower.` via the `_vt` attribute.
    """
    def __init__(self, cfg):
        super().__init__()
        self.patch = cfg.get("patch_size", PATCH)
        hdim = cfg.get("vt_hidden_size", 1024)
        qkvh = cfg.get("qkv_hidden_size", 1536)
        inter = cfg.get("vt_intermediate_size", 4096)
        layers = cfg.get("vt_num_hidden_layers", 27)
        heads = cfg.get("vt_num_attention_heads", 12)
        self.merge = tuple(cfg.get("merge_kernel_size", [2, 2]))[0]
        self._vt = nn.Module()
        self._vt.patch_embed = PatchEmbed(hdim)
        self._vt.encoder = _Encoder(hdim, qkvh, inter, layers, heads)
        self.rope = Rope2D(qkvh // heads)

    @property
    def patch_embed(self):
        return self._vt.patch_embed

    @property
    def encoder(self):
        return self._vt.encoder

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> List[torch.Tensor]:
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=next(self.parameters()).dtype)
        grid_thws = grid_thws.to(device)
        grid_hw = [(int(t), int(h), int(w)) for t, h, w in grid_thws.tolist()]
        # patch embed + pos
        feats = self.patch_embed(pixel_values, [(h, w) for _, h, w in grid_hw])
        # cu_seqlens + per-image RoPE freqs
        cu = [0]
        cos_parts, sin_parts = [], []
        for (t, h, w) in grid_hw:
            n = t * h * w
            cu.append(cu[-1] + n)
            c, s = self.rope.get(h, w)
            cos_parts.append(c.repeat(t, 1).to(pixel_values.device))
            sin_parts.append(s.repeat(t, 1).to(pixel_values.device))
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
        cos = torch.cat(cos_parts, dim=0)
        sin = torch.cat(sin_parts, dim=0)
        max_seqlen = max(cu[i + 1] - cu[i] for i in range(len(cu) - 1))

        x = feats
        for blk in self.encoder.blocks:
            x = blk(x, (cos, sin), cu_seqlens, max_seqlen)
        x = self.encoder.final_layernorm(x)
        # split + 2x2 sd2_tpool merge
        outs, ofs = [], 0
        for (t, h, w) in grid_hw:
            n = t * h * w
            f = x[ofs:ofs + n].view(t, h // 2, 2, w // 2, 2, -1)
            f = f.permute(0, 1, 3, 2, 4, 5).mean(dim=0).view(-1, 4, x.shape[-1])
            outs.append(f); ofs += n
        return outs


def load_moonvit_from_safetensors(safetensors_path: str, vision_config: dict,
                                  device="cpu", dtype=torch.bfloat16) -> "MoonViTV2":
    from safetensors.torch import load_file
    model = MoonViTV2(vision_config)
    sd = load_file(safetensors_path)
    # checkpoint keys are canonical-prefixed `vision_tower.<...>`; bind to `_vt` subtree.
    remap = {}
    for k in sd.keys():
        assert k.startswith("vision_tower."), f"unexpected checkpoint root: {k}"
        remap["_vt." + k[len("vision_tower."):]] = sd[k]
    missing, unexpected = model.load_state_dict(remap, strict=False)
    unexpected = [u for u in unexpected]
    missing = [m for m in missing
               if "rope" not in m and "_pos_time" not in m]  # non-persistent buffers
    if unexpected:
        raise RuntimeError(f"unexpected state keys: {unexpected[:6]}")
    real_missing = [m for m in missing if m in set(".".join(m.split('.')) for m in remap.values())]
    if real_missing:
        raise RuntimeError(f"missing weights: {real_missing[:6]}")
    model.to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model
