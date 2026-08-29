"""vision_adapter/data/cauldron.py — Cauldron subset pull (stub).

Extracted from master:modal_pipeline.py::cauldron_pull.  Contract:
  cauldron_manifest.jsonl — raw Cauldron pull before the sampled recipe.
  Two constants document the parallelism intent (not exercised in dry_run):
    N_DL=6   parallel parquet downloads
    N_SAVE=12 parallel PNG encodes
"""

from __future__ import annotations

import hashlib
from pathlib import Path

N_DL = 6  # parallel parquet downloads (HF hub)
N_SAVE = 12  # parallel PNG encodes (CPU-bound)


def pull_cauldron(
    backend=None,
    out_dir: Path | str | None = None,
    max_rows: int = 54000,
    dry_run: bool = False,
    revision: str | None = None,
) -> list[dict]:
    """Pull permissive Cauldron subsets and return row dicts.

    Each row is {"images": [paths], "texts": [{"user":..., "assistant":...}], "subset": str}.
    For dry_run (tests) returns ``max_rows`` fake rows without touching HF.
    Otherwise attempts ``datasets.load_dataset("HuggingFaceM4/the_cauldron")``
    punted to a stub (0 rows) if datasets is absent — dry_run is what tests cover.
    """
    if dry_run:
        rows: list[dict] = []
        for i in range(max_rows):
            rel = f"cauldron/fake_{i:06d}.png"
            sha = hashlib.sha1(rel.encode()).hexdigest()[:20]
            rows.append(
                {
                    "images": [f"embeddings/{sha}.pt"],
                    "texts": [{"user": f"cauldron user {i}", "assistant": f"cauldron assistant {i}"}],
                    "subset": "cauldron",
                }
            )
        return rows

    # Non-dry_run: attempt real HF pull (optional dep). Stub to 0 rows if unavailable.
    try:
        from datasets import load_dataset  # type: ignore

        _ = load_dataset  # silence linter
        _ = revision  # pinned revision would be forwarded here
        # Real implementation would iterate DOC/CONV subsets, download parquet
        # shards with N_DL concurrency and save PNGs with N_SAVE workers,
        # writing cauldron_manifest.jsonl and cauldron_done.txt checkpoints.
        return []
    except Exception:
        return []
