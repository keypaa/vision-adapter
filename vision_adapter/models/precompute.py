"""vision_adapter/models/precompute.py — staged-CLI precompute entrypoint.

Delegates to the real MoonViT pipeline in precompute_colab (same _emb_key
contract, same preprocess.collate_images); this module is the thin
backend-aware wrapper the CLI calls. Local-filesystem backends only — remote
backends (Modal Volume) go through modal_pipeline.py upstream.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _emb_key(image_path: str) -> str:
    """Volume-relative embedding key (shared with precompute_colab.moonvit helpers)."""
    rel = image_path.split("/images/", 1)[-1] if "/images/" in image_path else image_path
    return hashlib.sha1(rel.encode()).hexdigest()[:20] + ".pt"


def run_precompute(
    backend=None,
    data_dir: Path | str | None = None,
    patch_cap: int = 262144,
    device: str = "cuda",
    revision: str | None = None,
) -> None:
    """Run MoonViT precompute over <data_dir>/images into <data_dir>/embeddings.

    Fixed 2026-09-30 (was a validation-only stub printing "ok"): now delegates
    to precompute_colab.run with the data_dir roots. Requires a
    local-filesystem backend (LocalBackend.root); anything else raises loudly
    instead of silently doing nothing.
    """
    if backend is None:
        raise ValueError("backend is required (DataBackend)")
    if data_dir is None:
        raise ValueError("data_dir is required")
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"data_dir does not exist: {p}")
    if patch_cap <= 0:
        raise ValueError(f"patch_cap must be >0, got {patch_cap}")
    if device not in ("cuda", "cpu", "mps"):
        raise ValueError(f"unsupported device {device!r}")
    if getattr(backend, "root", None) is None:
        raise ValueError(
            "run_precompute needs a local-filesystem backend (LocalBackend with "
            ".root) — remote backends run precompute via modal_pipeline.py, not "
            "the staged CLI"
        )
    _ = revision  # reserved for HF revision pin forwarded to moonvit weight fetch
    from vision_adapter.models import precompute_colab as _pc

    _pc.run(
        images_root=str(p / "images"),
        out_root=str(p / "embeddings"),
        batch_patches=patch_cap,
        device=device,
    )
    return None
