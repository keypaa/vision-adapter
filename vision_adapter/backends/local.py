"""vision_adapter/backends/local.py — filesystem-backed DataBackend."""
from __future__ import annotations

from pathlib import Path

import torch


class LocalBackend:
    """Local filesystem backend: root is a directory (e.g. ./data or emb_cache)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _p(self, key: str) -> Path:
        return self.root / key

    def list_embeddings(self, prefix: str) -> list[str]:
        # prefix like "embeddings/" — return sorted keys relative to root
        hits = [str(p.relative_to(self.root)) for p in self.root.glob(f"{prefix}*.pt")]
        return sorted(hits)

    def read_embedding(self, key: str) -> torch.Tensor:
        return torch.load(self._p(key), map_location="cpu", weights_only=True)

    def write_embedding(self, key: str, tensor: torch.Tensor) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, p)

    def exists(self, path: str) -> bool:
        return self._p(path).exists()

    def read_bytes(self, path: str) -> bytes:
        return self._p(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
