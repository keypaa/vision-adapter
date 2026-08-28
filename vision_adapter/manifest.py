"""vision_adapter/manifest.py — versioned train-manifest I/O.

Lean step 3 of refactor/discipline: replaces the two scattered manifest paths
(at grok_probe_qwen.py: fetch_manifest + build_epoch_plan and modal_train.py:
EmbSFT) with one header-first JSONL contract. Pluggable into a future pipeline
via `write_manifest_with_header`; on read the header is best-effort (old
files without it still parse).

File format (v1):
  line 0: {"type":"manifest_header","manifest_version":1,"git_sha":...,"seeds":{...},
           "created_at":...,"upstream":{...},"shard_set_hash":...,"row_count":N,"tags":...}
  line 1..N: {"emb":"embeddings/<sha>.pt","user":...,"assistant":...,"g":...}

Reading tolerates either case (header present or absent) and exposes the header
via read_manifest_header() without consuming the rows. Rows iterator skips any
header-like row (type == manifest_header).

The ORDER BY determinism fix from docs/research/best-practices.md lives here:
`SELECT ... FROM read_parquet(...) WHERE ... ORDER BY image LIMIT 54000` —
without ORDER BY the shard's row order (parquet write order) drifts between
rebuilds and `random.seed(0)` cannot save you.

Keep this file dependency-light: only stdlib + config helpers.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from vision_adapter.config import get_git_sha


MANIFEST_VERSION = 1

# Upstream revisions recorded in the header (same keys the provenance audit cited).
# Callers that know the exact Hugging Face revisions may pass them in; defaults
# stay human-readable and document the gap that pre-existed this file.
DEFAULT_UPSTREAM: dict[str, str] = {
    "agentic_source": "0xSero/glm-vision-sft-mix@refs/convert/parquet",
    "cauldron_source": "HuggingFaceM4/the_cauldron",  # etl stage pulls via huggingface download
}


def shard_set_hash(shard_files: list[str] | None) -> str | None:
    """Stable hash of the shard-file set (order-independent)."""
    if not shard_files:
        return None
    h = hashlib.sha256()
    for s in sorted(shard_files):
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class ManifestHeader:
    manifest_version: int = MANIFEST_VERSION
    git_sha: str = field(default_factory=get_git_sha)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    seeds: dict[str, int] = field(default_factory=lambda: {"python": 0, "numpy": 0, "torch": 0})
    upstream: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UPSTREAM))
    shard_set_hash: str | None = None
    row_count: int | None = None
    tags: dict[str, Any] | None = None  # free-form: e.g. {"sample_size": 120000}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = "manifest_header"
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ManifestHeader":
        return cls(
            manifest_version=int(d.get("manifest_version", MANIFEST_VERSION)),
            git_sha=str(d.get("git_sha", "unknown")),
            created_at=str(d.get("created_at", "")),
            seeds=dict(d.get("seeds", {})),
            upstream=dict(d.get("upstream", {})),
            shard_set_hash=d.get("shard_set_hash"),
            row_count=d.get("row_count"),
            tags=d.get("tags"),
        )


def write_manifest_with_header(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    seeds: dict[str, int] | None = None,
    upstream: dict[str, str] | None = None,
    shard_files: list[str] | None = None,
    tags: dict[str, Any] | None = None,
) -> Path:
    """Write a header-first manifest JSONL to `path` (atomic via tmp+replace).

    Rows are the usual {emb, user, assistant, g} dicts. The header captures
    provenance that makes a rebuild provably identical (git SHA, seeds,
    upstream revisions, shard set hash, row count, timestamp).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ManifestHeader(
        git_sha=get_git_sha(),
        seeds=seeds if seeds is not None else {"python": 0, "numpy": 0, "torch": 0},
        upstream=upstream if upstream is not None else dict(DEFAULT_UPSTREAM),
        shard_set_hash=shard_set_hash(shard_files),
        row_count=len(rows),
        tags=tags,
    )
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(header.to_dict(), separators=(",", ":")) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(out)
    return out


def read_manifest_header(path: str | Path) -> ManifestHeader | None:
    """Peek the header (line 0) if present; None if absent or unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open() as f:
            first = f.readline()
        if not first.strip():
            return None
        obj = json.loads(first)
        if obj.get("type") != "manifest_header":
            return None
        return ManifestHeader.from_dict(obj)
    except Exception:
        return None


def iter_manifest_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield data rows, skipping a leading header row if present."""
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if obj.get("type") == "manifest_header":
                continue
            yield obj


def load_manifest(path: str | Path) -> tuple[list[dict[str, Any]], ManifestHeader | None]:
    """Load rows + header (None if legacy file without one)."""
    header = read_manifest_header(path)
    rows = list(iter_manifest_rows(path))
    return rows, header
