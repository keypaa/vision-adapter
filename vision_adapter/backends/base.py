"""vision_adapter/backends/base.py — DataBackend protocol + factory."""
from __future__ import annotations

from typing import Protocol

import torch


class DataBackend(Protocol):
    """Abstract I/O for embeddings and generic paths.

    Local and Modal share the same surface so every stage takes DataBackend
    instead of branching on `if modal`.
    """

    def list_embeddings(self, prefix: str) -> list[str]:
        """List embedding keys with given prefix (e.g. "embeddings/")."""
        ...

    def read_embedding(self, key: str) -> torch.Tensor:
        """Read one embedding tensor (bf16/float, shape [n_vis, 4096])."""
        ...

    def write_embedding(self, key: str, tensor: torch.Tensor) -> None:
        """Write one embedding tensor (atomic where possible)."""
        ...

    def exists(self, path: str) -> bool:
        """Whether path exists."""
        ...

    def read_bytes(self, path: str) -> bytes:
        """Read raw bytes (for generic files)."""
        ...


def get_backend(name: str, **kwargs) -> DataBackend:
    """Factory: "local" | "modal". Extra kwargs forwarded to backend ctor.

    - local: kwargs `root` (Path | str) required.
    - modal: kwargs `volume_name` optional.
    """
    if name == "local":
        from vision_adapter.backends.local import LocalBackend

        return LocalBackend(**kwargs)  # type: ignore[return-value]
    if name == "modal":
        from vision_adapter.backends.modal import ModalBackend

        return ModalBackend(**kwargs)  # type: ignore[return-value]
    raise ValueError(f"unknown backend {name!r} — expected 'local' or 'modal'")
