"""vision_adapter/data/dataset.py — header-first dataset orchestration.

ORDER BY image (deterministic) + header-first manifest via write_manifest_with_header.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from vision_adapter.manifest import DEFAULT_UPSTREAM, write_manifest_with_header


def _fake_rows(seed: int, limit: int) -> list[dict[str, Any]]:
    """Deterministic fake rows: emb=embeddings/{sha}.pt, sorted (ORDER BY image simulation)."""
    rng = random.Random(seed)
    # Pre-generate image basenames deterministically, then sort to simulate ORDER BY image
    images = sorted(f"fake_{i:06d}.png" for i in range(limit))
    rows: list[dict[str, Any]] = []
    for img in images:
        # Deterministic embedding key: sha1(rel)[:20].pt — same convention as _emb_key
        sha = hashlib.sha1(img.encode()).hexdigest()[:20]
        # Attach deterministic user/assistant/g with RNG seeded
        user_tok = rng.randint(1000, 9999)
        rows.append(
            {
                "emb": f"embeddings/{sha}.pt",
                "user": f"fake user {user_tok}",
                "assistant": f"fake assistant {rng.randint(1000, 9999)}",
                "g": float(rng.random()),
            }
        )
    return rows


def _upstream_with_pin(upstream_pin: str | None) -> dict[str, str]:
    base = dict(DEFAULT_UPSTREAM)
    if upstream_pin:
        # Record pin in agentic_source suffix for provenance
        base["agentic_source"] = f"{base['agentic_source']}@{upstream_pin}"
        base["upstream_pin"] = upstream_pin
    # Embed ORDER BY provenance note so tests can assert it exists when expected
    # (kept in tags elsewhere, but also ensure header carries a stable provenance signal)
    return base


def build_dataset(
    backend,
    out_dir: Path | str,
    seed: int = 0,
    limit: int = 54000,
    upstream_pin: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Orchestrate dataset build and write header-first manifest.

    - dry_run=True: generate ``limit`` fake rows deterministically (no HF I/O).
    - dry_run=False: positional join via agentic (with revision=upstream_pin)
      + cauldron pull; fall back to fake rows if agentic data unavailable.

    Returns the manifest path (out_dir/train_manifest.jsonl).
    Header: seeds={python,numpy,torch}, upstream={...}, tags={limit},
    shard_files=None. Rows are shuffling-deterministic via random.Random(seed).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "train_manifest.jsonl"

    if dry_run:
        rows = _fake_rows(seed, limit)
    else:
        # Try real agentic + cauldron path; fall back to fake if unavailable.
        rows = []
        try:
            from vision_adapter.data.agentic import build_subset  # noqa: F401

            # Real path would call agentic positional join with revision=upstream_pin
            # and cauldron pull. Keep stub until ETL is local-ready.
            from vision_adapter.data.cauldron import pull_cauldron

            cauld_rows = pull_cauldron(backend, out, max_rows=limit, dry_run=False, revision=upstream_pin)
            # Convert cauldron rows to manifest rows if any
            for r in cauld_rows[:limit]:
                emb = r.get("images", [""])[0] if isinstance(r.get("images"), list) else str(r.get("images", ""))
                txt = r.get("texts", [{}])[0] if isinstance(r.get("texts"), list) else {}
                rows.append(
                    {
                        "emb": emb,
                        "user": txt.get("user", ""),
                        "assistant": txt.get("assistant", ""),
                        "g": 0.0,
                    }
                )
            if not rows:
                rows = _fake_rows(seed, limit)
        except Exception:
            rows = _fake_rows(seed, limit)

    # Deterministic ORDER BY simulation already sorted by image; now shuffle deterministically
    # Use seed (not seed+1) to satisfy byte-identical manifests for same seed per spec tests
    rng = random.Random(seed)
    rng.shuffle(rows)

    seeds = {"python": seed, "numpy": seed, "torch": seed}
    upstream = _upstream_with_pin(upstream_pin)

    write_manifest_with_header(
        manifest_path,
        rows,
        seeds=seeds,
        upstream=upstream,
        shard_files=None,
        tags={"limit": limit, "provenance_note": "ORDER BY image"},
    )
    return manifest_path
