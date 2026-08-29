"""vision_adapter/models/precompute.py — shared MoonViT precompute wrapper.

Shared _emb_key: sha1(rel)[:20].pt where rel is volume-relative logical path
(agentic/foo.png, cauldron/foo.png) so Modal and local produce identical keys.
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
    """Validate args and (in prod) run moonvit + preprocess over the corpus.

    Stub: validates backend/data_dir exist and would call moonvit.py + preprocess.py
    machinery. For tests, this only validates args.
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
    _ = revision  # reserved for HF revision pin forwarded to moonvit weight fetch
    _ = _emb_key  # keep shared helper live for importers
    return None
