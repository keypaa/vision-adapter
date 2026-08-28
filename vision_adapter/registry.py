"""vision_adapter/registry.py — one local file for run comparison.

Lean step 5: closes the "Experiment registry + enriched logging" gap from
best-practices §Checklist without a new service. One JSONL file per-run
summaries: `Configuration | sec/step | samples/sec | tokens/sec | VRAM |
Relative` (see `experiment.csv` in Marin's ladder, Fig. 4 of that issue).

Written best-effort at run_end; the trainer's hot loop never depends on it.
Each entry is one JSON object with run_id correlation to the per-step
config_header + run_end already in the trainer's JSONL stream.

No modal dependency. Keep free of heavy imports.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

REGISTRY_PATH = "runs.jsonl"  # default relative to the log directory


def registry_entry(
    *,
    run_id: str | None,
    git_sha: str | None,
    config: dict[str, Any] | None = None,
    manifest_sha256: str | None = None,
    manifest_rows: int | None = None,
    shard_set_hash: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    dtype: str | None = None,
    step_ms: float | None = None,
    tokens_per_sec: float | None = None,
    samples_per_sec: float | None = None,
    peak_gib: float | None = None,
    wall_min: float | None = None,
    final_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one registry row (JSON-serialisable). Callers fill what they have."""
    row: dict[str, Any] = {
        "run_id": run_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha,
        "config": config or {},
        "manifest_sha256": manifest_sha256,
        "manifest_rows": manifest_rows,
        "shard_set_hash": shard_set_hash,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "step_ms": step_ms,
        "tokens_per_sec": tokens_per_sec,
        "samples_per_sec": samples_per_sec,
        "peak_gib": peak_gib,
        "wall_min": wall_min,
        "final_loss": final_loss,
    }
    if extra:
        row.update(extra)
    # drop Nones so the JSONL stays compact
    return {k: v for k, v in row.items() if v is not None}


def append_registry(
    path: str | Path,
    row: dict[str, Any],
) -> None:
    """Append one JSON object (one line) to `path` — atomic append, buffering=1."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", buffering=1) as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_registry(path: str | Path) -> list[dict[str, Any]]:
    """Read all rows (tolerates missing file -> [])."""
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    with p.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return out
