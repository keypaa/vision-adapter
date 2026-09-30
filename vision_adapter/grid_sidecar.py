"""vision_adapter/grid_sidecar.py — measured MoonViT geometry, keyed by emb.

Why this exists (audit 2026-09-30): the precomputed corpus stores ``n_vis``
but no grid, so training fell back to ``grid_for_nvis`` — a squarest-even
factorization that INVERTS orientation (a 56x26 portrait, n_vis=364, was fed
to mRoPE as 28x52 landscape; x3.0 median aspect error, 200/200 rows wrong).
The grid is a deterministic function of the image dims under the preprocess
contract, so it is recoverable without re-running the ViT: measured against
the live key index, 400/400 rows matched exactly.

Rows lacking a sidecar entry keep the synthetic fallback, but the caller gets
a per-row source label so a run card can declare ``grid_source`` and refuse
to compare curves across the two regimes.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable

from vision_adapter.core import grid_for_nvis


def build_sidecar_rows(
    batches: Iterable[Any],
    key_fn,
    dims_fn,
) -> list[dict[str, Any]]:
    """Turn streamed corpus batches into ``{emb, grid_thw}`` rows.

    ``key_fn(group, basename) -> emb`` gives the embedding key for a file.
    ``dims_fn(group, basename) -> (w, h) | None`` gives its pixel size; a
    ``None`` (unreadable, not yet downloaded) omits the row rather than
    guessing a grid.
    """
    from vision_adapter.core import grid_from_dims

    rows = []
    for batch in batches:
        names = batch.column("filename").to_pylist()
        blobs = batch.column("image").to_pylist()
        for name, blob in zip(names, blobs):
            group, _, base = str(name).rpartition("/")
            dims = dims_fn(group, base)
            if dims is None:
                continue
            grid = grid_from_dims(int(dims[0]), int(dims[1]))
            rows.append({"emb": key_fn(group, base), "grid_thw": grid.tolist()})
        del batch
    return rows


class GridSidecar:
    """emb key -> measured ``[1, gh, gw]``, with a per-row fallback label."""

    def __init__(self, path: str | Path | None):
        self.data: dict[str, list[int]] = {}
        if path is None:
            return
        p = Path(path)
        if not p.is_file():
            return
        try:
            raw = json.loads(p.read_text())
        except Exception:
            return
        self.data = {k: list(v) for k, v in raw.items()}

    def get(self, emb: str) -> list[int] | None:
        return self.data.get(emb)

    def __len__(self) -> int:
        return len(self.data)

    def grids_for(
        self, embs: list[str], n_vis: list[int], merge_size: int = 2
    ) -> tuple[list[list[int]], list[str]]:
        """Return ``(grids, sources)`` aligned with ``embs`` / ``n_vis``.

        ``sources[i]`` is ``"measured"`` when the sidecar has the row, else
        ``"synthetic"`` from ``grid_for_nvis``. A measured grid whose
        placeholder count disagrees with ``n_vis`` raises — that is a data
        bug, not something to paper over.
        """
        grids: list[list[int]] = []
        sources: list[str] = []
        for i, emb in enumerate(embs):
            measured = self.get(emb)
            if measured is None:
                grids.append(grid_for_nvis(n_vis[i], merge_size).tolist())
                sources.append("synthetic")
                continue
            total = measured[0] * measured[1] * measured[2]
            if total % (merge_size**2) or total // (merge_size**2) != n_vis[i]:
                raise ValueError(
                    f"{emb}: grid_thw {measured} mismatch: gives "
                    f"{total // (merge_size**2)} placeholders but n_vis={n_vis[i]}"
                )
            grids.append(list(measured))
            sources.append("measured")
        return grids, sources


def image_dims(blob: bytes) -> tuple[int, int] | None:
    """(w, h) of encoded image bytes, or None if PIL cannot read them."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(blob)) as im:
            return im.size
    except Exception:
        return None
