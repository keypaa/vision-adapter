#!/usr/bin/env python3
"""Rebuild a val manifest disjoint by emb from train.

Usage:
    python scripts/rebuild_val_disjoint.py \
        --train /tmp/opencode/manifest_audit_cache/train_manifest.jsonl \
        --val-in /tmp/opencode/manifest_audit_cache/train_manifest_val.jsonl \
        --val-out /tmp/opencode/train_manifest_val_disjoint.jsonl

Reads both manifests (header tolerated), drops val rows whose emb key appears
in train (see vision_adapter.manifest.disjoint_val_rows), writes a header-first
manifest with provenance tags. Prints kept/dropped counts + g distribution
before/after. Pushing the result to HF is a separate, human-run step
(see docs/research/qwen-injection-next.md NEXT-6).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_adapter.manifest import (  # noqa: E402
    disjoint_val_rows,
    load_manifest,
    write_manifest_with_header,
)


def _read_raw(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val-in", required=True)
    ap.add_argument("--val-out", required=True)
    args = ap.parse_args()

    train_rows, _ = load_manifest(args.train)
    val_rows, _ = load_manifest(args.val_in)
    kept = disjoint_val_rows(train_rows, val_rows)
    dropped = len(val_rows) - len(kept)

    before = Counter(r.get("g", "?") for r in val_rows)
    after = Counter(r.get("g", "?") for r in kept)
    print(f"train rows : {len(train_rows)}")
    print(f"val in     : {len(val_rows)} g={dict(before)}")
    print(f"dropped    : {dropped} (emb seen in train)")
    print(f"val out    : {len(kept)} g={dict(after)}")
    val_embs = [r["emb"] for r in kept]
    print(f"dup emb inside new val: {len(val_embs) - len(set(val_embs))}")
    assert not (set(val_embs) & {r["emb"] for r in train_rows}), "overlap remains!"

    write_manifest_with_header(
        args.val_out,
        kept,
        tags={
            "derived_from": "train_manifest_val.jsonl",
            "method": "disjoint by emb vs train_manifest.jsonl",
            "val_in_rows": len(val_rows),
            "dropped_overlap": dropped,
            "date": "2026-09-30",
        },
    )
    print(f"wrote {args.val_out} (header-first, disjoint by emb: verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
