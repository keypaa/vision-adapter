"""Kimi-K3 MoonViT image preprocessing contract.

Resize (navit_resize_image) -> zero-pad to multiple of 28 -> normalize -> patchify.
Deterministic; no I/O beyond the input PIL image.
"""

import math

import numpy as np
import torch
from PIL import Image

PATCH = 14
MAX_PATCHES = 65536
MAX_SIDE = 512 * 14  # 7168
PAD_TO = 28

_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _resize_size(w: int, h: int) -> tuple[int, int]:
    patches = (w // PATCH) * (h // PATCH)
    scale = min(
        1.0,
        math.sqrt(MAX_PATCHES / patches) if patches > 0 else 1.0,
        MAX_SIDE / w,
        MAX_SIDE / h,
    )
    new_w = min(int(w * scale), MAX_SIDE)
    new_h = min(int(h * scale), MAX_SIDE)
    return max(new_w, PATCH), max(new_h, PATCH)


def _pad_to_28(n: int) -> int:
    return n + (PAD_TO - n % PAD_TO) % PAD_TO


def process_image(pil_img: Image.Image) -> dict:
    """Preprocess one PIL image -> patch tokens + grid_thws."""
    img = pil_img.convert("RGB")
    w, h = img.size

    new_w, new_h = _resize_size(w, h)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.BICUBIC)

    w_pad, h_pad = _pad_to_28(new_w), _pad_to_28(new_h)

    x = np.asarray(img, dtype=np.float32)  # (H, W, 3)
    x = (x / 255.0 - _MEAN) * (1.0 / _STD)  # normalize

    canvas = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
    canvas[:new_h, :new_w, :] = x  # constant-0 pad, right/bottom

    t = torch.from_numpy(canvas).permute(2, 0, 1)  # (3, H_pad, W_pad)
    gh, gw = h_pad // PATCH, w_pad // PATCH
    patches = (
        t.unfold(1, PATCH, PATCH)
        .unfold(2, PATCH, PATCH)  # (3, gh, gw, 14, 14)
        .permute(1, 2, 0, 3, 4)  # (gh, gw, 3, 14, 14), row-major h-major
        .reshape(gh * gw, 3, PATCH, PATCH)
        .contiguous()
    )
    grid = torch.tensor([[1, gh, gw]], dtype=torch.long)
    return {"pixel_values": patches, "grid_thws": grid}


def collate_images(pil_images: list) -> dict:
    """Pack multiple PIL images into one pixel_values + one grid row per image."""
    outs = [process_image(im) for im in pil_images]
    pixel_values = torch.cat([o["pixel_values"] for o in outs], dim=0)
    grid_thws = torch.cat([o["grid_thws"] for o in outs], dim=0)
    return {"pixel_values": pixel_values, "grid_thws": grid_thws}
