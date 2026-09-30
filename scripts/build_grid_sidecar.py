#!/usr/bin/env python3
"""Build the grid sidecar (emb key -> measured MoonViT geometry) from the corpus.

Streams the image corpus parquet in bounded row groups — never materialises a
whole shard — reads each image's pixel dims, and derives the true (1, gh, gw)
grid under the preprocess contract. RAM stays ~one batch wide.

Usage:
    python scripts/build_grid_sidecar.py --out grid_sidecar.json \
        --shards /tmp/opencode/img_sample/data/train-00000-of-00017.parquet

Optional --verify joins the key index and reports how many stored n_vis the
derived grids reproduce (the audit measured 400/400).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_adapter.grid_sidecar import image_dims  # noqa: E402

BATCH = 64


def _emb_key(group: str, basename: str) -> str:
    rel = f"{group}/{basename}" if group else basename
    return f"embeddings/{hashlib.sha1(rel.encode()).hexdigest()[:20]}.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = every row")
    ap.add_argument("--verify-index", help="key_index_cache json to cross-check n_vis")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    from vision_adapter.core import grid_from_dims, placeholder_count

    grids: dict[str, list[int]] = {}
    seen = unreadable = 0
    for shard in args.shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(
            batch_size=BATCH, columns=["filename", "image"]
        ):
            for name, blob in zip(
                batch.column("filename").to_pylist(),
                batch.column("image").to_pylist(),
            ):
                if args.limit and seen >= args.limit:
                    break
                seen += 1
                group, _, base = str(name).rpartition("/")
                dims = image_dims(blob)
                if dims is None:
                    unreadable += 1
                    continue
                grids[_emb_key(group, base)] = grid_from_dims(*dims).tolist()
            del batch
            if args.limit and seen >= args.limit:
                break
        print(f"{Path(shard).name}: {len(grids)} entries so far", flush=True)

    Path(args.out).write_text(json.dumps(grids, separators=(",", ":")))
    print(f"seen={seen} unreadable={unreadable} wrote {len(grids)} -> {args.out}")

    if args.verify_index:
        from vision_adapter.data.stream import load_key_index

        index, ok = load_key_index(args.verify_index)
        assert ok, "key index unreadable"
        match = mismatch = skipped = 0
        for emb, grid in grids.items():
            loc = index.get(emb)
            if not loc or len(loc) != 3:
                skipped += 1
                continue
            if placeholder_count(grid, 2) == loc[2]:
                match += 1
            else:
                mismatch += 1
        print(
            f"verify against n_vis index: match={match} mismatch={mismatch} "
            f"not-in-index={skipped} ({100 * match / max(1, match + mismatch):.1f}%)"
        )
        if mismatch:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
