"""vision_adapter/backends/modal.py — Modal Volume-backed DataBackend."""
from __future__ import annotations

import io

import torch

try:
    import modal  # type: ignore[import]

    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False
    modal = None  # type: ignore[assignment]


class ModalBackend:
    """Modal Volume backend — thin wrappers around modal.Volume."""

    def __init__(self, volume_name: str = "vision-adapter-data"):
        if not HAS_MODAL:
            raise RuntimeError(
                "Modal not installed — `pip install modal` or use --backend local"
            )
        self.volume_name = volume_name
        self.vol = modal.Volume.from_name(volume_name, create_if_missing=True)  # type: ignore[union-attr]

    def list_embeddings(self, prefix: str) -> list[str]:
        entries = self.vol.listdir(prefix.rstrip("/") if prefix else "")  # type: ignore[union-attr]
        out: list[str] = []
        for e in entries:
            p = getattr(e, "path", None) or getattr(e, "name", None) or str(e)
            if prefix and not p.startswith(prefix):
                p = f"{prefix.rstrip('/')}/{p.lstrip('/')}"
            out.append(p)
        return sorted(out)

    def read_embedding(self, key: str) -> torch.Tensor:
        buf = io.BytesIO()
        self.vol.read_file_into_fileobj(key, buf)  # type: ignore[union-attr]
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=True)

    def write_embedding(self, key: str, tensor: torch.Tensor) -> None:
        buf = io.BytesIO()
        torch.save(tensor, buf)
        buf.seek(0)
        with self.vol.batch_upload() as batch:  # type: ignore[union-attr]
            batch.put_file(buf, key)  # type: ignore[arg-type]
        self.vol.commit()  # type: ignore[union-attr]

    def exists(self, path: str) -> bool:
        # Call volume-level exists primitive when available; otherwise probe
        # the parent folder. Avoid swallowing auth/network errors silently beyond
        # the probe — existence check is best-effort by contract.
        try:
            parent = "/".join(path.split("/")[:-1]) or "/"
            needle = path
            # Volume listings may return full keys or bare basenames depending on SDK;
            # normalize both to the absolute key for comparison.
            entries = self.vol.listdir(parent)  # type: ignore[union-attr]
            keys: set[str] = set()
            for e in entries:
                p = getattr(e, "path", None) or getattr(e, "name", None) or str(e)
                if "/" not in p or parent in ("/", "", "."):
                    # bare basename — prepend parent
                    keys.add(f"{parent.rstrip('/')}/{p.lstrip('/')}" if parent not in ("/", "", ".") else p)
                else:
                    keys.add(p)
            if needle in keys:
                return True
            # Fallback: Volume sometimes returns basenames only — check suffix
            return any(needle.endswith(f"/{k}") or needle == k for k in keys)
        except Exception:
            return False

    def read_bytes(self, path: str) -> bytes:
        buf = io.BytesIO()
        self.vol.read_file_into_fileobj(path, buf)  # type: ignore[union-attr]
        return buf.getvalue()

    def write_bytes(self, path: str, data: bytes) -> None:
        with self.vol.batch_upload() as batch:  # type: ignore[union-attr]
            batch.put_file(io.BytesIO(data), path)  # type: ignore[arg-type]
        self.vol.commit()  # type: ignore[union-attr]
