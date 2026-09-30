#!/usr/bin/env python3
"""Backfill grid_thw into a train manifest from the image corpus.

The MoonViT grid is a deterministic function of the image dims under the
preprocess contract (resize -> pad-to-28 -> 14px patches), so it is
recoverable without re-running the ViT: verified 400/400 live rows against
the n_vis stored in the embedding key index.

Why it matters: the synthetic grid_for_nvis stand-in INVERTS orientation —
a 56x26 portrait (n_vis=364) came out as 28x52 landscape, x3.0 median
aspect error over 200 live rows. That is the worst case for exactly the
UI/OCR/chart data this trains on.

The image corpus is streamed in bounded batches, never loaded whole, so RAM
stays ~one batch wide (measured 1.15 GB peak, essentially the key index).

Usage:
    python scripts/backfill_grid_thw.py \
        --manifest train_manifest.jsonl \
        --out train_manifest_grids.jsonl \
        --shards /data/images/train-*.parquet \
        --verify-index key_index_cache.<hash>.json

Writes a header-first manifest; rows that could not be matched keep no
grid_thw and are counted in the header tags, so coverage is auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_adapter.manifest import (  # noqa: E402
    grid_thw_for_row,
    load_manifest,
    write_manifest_with_header,
)

BATCH = 64


def emb_key(group: str, basename: str) -> str:
    """Same key the precompute uses: sha1 of the volume-relative logical path."""
    rel = f"{group}/{basename}" if group else basename
    return f"embeddings/{hashlib.sha1(rel.encode()).hexdigest()[:20]}.pt"


def image_dims(blob: bytes):
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(blob)) as im:
            return im.size
    except Exception:
        return None


def scan_corpus(shards, wanted, log_every=20000):
    """emb key -> grid_thw, for the keys the manifest actually references.

    Streams each shard in bounded batches. Only the ``wanted`` keys are kept,
    so memory is bounded by the manifest size, not the corpus.
    """
    import pyarrow.parquet as pq

    from vision_adapter.core import grid_from_dims

    grids: dict[str, list[int]] = {}
    seen = unreadable = skipped = 0
    t0 = time.time()
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(
            batch_size=BATCH, columns=["filename", "image"]
        ):
            names = batch.column("filename").to_pylist()
            blobs = batch.column("image").to_pylist()
            for name, blob in zip(names, blobs):
                group, _, base = str(name).rpartition("/")
                key = emb_key(group, base)
                if key not in wanted:
                    skipped += 1
                    continue
                seen += 1
                dims = image_dims(blob)
                if dims is None:
                    unreadable += 1
                    continue
                grids[key] = grid_from_dims(dims[0], dims[1]).tolist()
            del batch, blobs
            if seen and seen % log_every < BATCH:
                rate = seen / max(1e-6, time.time() - t0)
                print(
                    f"  matched {seen} | grids {len(grids)} | "
                    f"skipped {skipped} | {rate:.0f} img/s",
                    flush=True,
                )
        print(f"[{Path(shard).name}] cumulative grids={len(grids)}", flush=True)
    return grids, seen, unreadable, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--verify-index", default=None,
                    help="key_index_cache json: cross-check derived grids vs stored n_vis")
    args = ap.parse_args()

    rows, header = load_manifest(args.manifest)
    wanted = {r["emb"] for r in rows if r.get("emb")}
    print(f"manifest: {len(rows)} rows, {len(wanted)} distinct emb keys", flush=True)
    already = sum(1 for r in rows if grid_thw_for_row(r) is not None)
    if already:
        print(f"note: {already} rows already carry grid_thw; they win", flush=True)

    print(f"scanning {len(args.shards)} shard(s) for those keys...", flush=True)
    grids, seen, unreadable, skipped = scan_corpus(args.shards, wanted)

    out_rows = []
    by_group = Counter()
    for r in rows:
        g = grid_thw_for_row(r) or grids.get(r.get("emb", ""))
        if g is not None:
            by_group[r.get("g", "?")] += 1
        new = {k: v for k, v in r.items() if k != "grid_thw"}
        if g is not None:
            new["grid_thw"] = g
        out_rows.append(new)

    have = sum(1 for r in out_rows if grid_thw_for_row(r) is not None)
    print(f"\ncoverage: {have}/{len(out_rows)} rows "
          f"({100 * have / max(1, len(out_rows)):.1f}%)")
    print(f"by group: {dict(by_group)}")
    print(f"unreadable images: {unreadable} | corpus rows skipped (unreferenced): {skipped}")

    rc = 0
    if args.verify_index:
        from vision_adapter.core import placeholder_count
        from vision_adapter.data.stream import load_key_index

        index, ok = load_key_index(args.verify_index)
        assert ok, "key index unreadable"
        match = mismatch = notfound = 0
        examples = []
        for r in out_rows:
            g = grid_thw_for_row(r)
            if g is None:
                continue
            loc = index.get(r["emb"])
            if not loc or len(loc) != 3:
                notfound += 1
                continue
            if placeholder_count(g, 2) == loc[2]:
                match += 1
            else:
                mismatch += 1
                if len(examples) < 5:
                    examples.append((r["emb"], g, loc[2]))
        rate = 100 * match / max(1, match + mismatch)
        print(f"verify vs stored n_vis: match={match} mismatch={mismatch} "
              f"not-in-index={notfound} ({rate:.1f}%)")
        for e in examples:
            print(f"  MISMATCH emb={e[0]} grid={e[1]} n_vis={e[2]}")
        if mismatch:
            print("ABORT: derived grids disagree with the stored n_vis — not writing")
            rc = 1

    if rc:
        return rc

    write_manifest_with_header(
        args.out,
        out_rows,
        tags={
            "derived_from": Path(args.manifest).name,
            "method": "grid_from_dims over the image corpus (preprocess contract)",
            "rows": len(out_rows),
            "with_grid": have,
            "without_grid": len(out_rows) - have,
            "by_group": dict(by_group),
            "unreadable_images": unreadable,
            "date": "2026-09-30",
        },
    )
    size_mb = os.path.getsize(args.out) / 2**20
    print(f"wrote {args.out} ({size_mb:.1f} MB) — grid_source will be "
          f"{'measured' if have == len(out_rows) else 'partial' if have else 'synthetic'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
