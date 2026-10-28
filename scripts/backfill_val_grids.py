#!/usr/bin/env python3
"""Backfill measured MoonViT grids for a disjoint validation manifest.

The validation manifest stores embedding keys but predates per-row ``grid_thw``.
This maps those keys to original images using the same SHA1 path contract as
precompute, derives the grid from image dimensions, and checks every grid
against the embedding shard's stored ``n_vis`` before writing.

Fetches only selected image rows from the images repository by parquet Range
reads; it does not download the 33.8 GiB repository. A local --images-dir can
be used instead. No geometry is inferred from n_vis, and no partial output is
written on failure.

Usage:
    python scripts/backfill_val_grids.py \\
        --manifest data/train_manifest_val_disjoint.jsonl \\
        --n 30 --min-vis-tokens 300 --max-vis-tokens 380 \\
        --out data/val_generation_measured.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _emb_key(relative_image_path: str) -> str:
    digest = hashlib.sha1(relative_image_path.encode()).hexdigest()[:20]
    return f"embeddings/{digest}.pt"


def find_images(images_dir: Path, wanted: set[str]) -> dict[str, Path]:
    """Find wanted embedding keys under the image root; reject ambiguous keys."""
    found: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        relative = path.relative_to(images_dir).as_posix()
        key = _emb_key(relative)
        if key not in wanted:
            continue
        if key in found:
            raise RuntimeError(f"multiple local images hash to validation key {key}")
        found[key] = path
    return found


def select_val_rows_by_n_vis(rows, index, n=30, seed=0,
                              min_n_vis=300, max_n_vis=380):
    """Deterministically select disjoint agentic rows by stored token count."""
    import random

    candidates = []
    missing_index = 0
    for row in rows:
        if row.get("g") != "agentic" or not row.get("emb"):
            continue
        loc = index.get(row["emb"])
        if loc is None or len(loc) < 3:
            missing_index += 1
            continue
        n_vis = int(loc[2])
        if min_n_vis <= n_vis <= max_n_vis:
            candidates.append(row)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max(0, n)], {
        "eligible": len(candidates), "selected": min(max(0, n), len(candidates)),
        "missing_index": missing_index,
    }


def image_sizes_from_paths(image_paths):
    from PIL import Image

    sizes = {}
    for key, path in image_paths.items():
        with Image.open(path) as image:
            sizes[key] = image.size
    return sizes


def fetch_remote_image_sizes(wanted, repo="keypa/vision-adapter-images"):
    """Range-scan filenames and fetch image bytes only for wanted keys."""
    import io

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from PIL import Image

    from vision_adapter.data.stream import RemoteShard, _fetch_range, _remote_size, rg_span

    files = sorted(
        name for name in HfApi().list_repo_files(repo, repo_type="dataset")
        if name.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"no parquet shards found in image repo {repo}")

    sizes = {}
    locations = {}
    for shard_no, shard in enumerate(files, 1):
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{shard}"
        remote = RemoteShard(url, _remote_size(url))
        pf = pq.ParquetFile(remote)
        md = pf.metadata
        for rgi in range(md.num_row_groups):
            lo, hi = rg_span(md, rgi, columns=("filename",))
            remote._span = (lo, _fetch_range(url, lo, hi - 1))
            names = pq.ParquetFile(remote).read_row_group(rgi, columns=["filename"]).column(
                "filename"
            ).to_pylist()
            hits = []
            for row_idx, filename in enumerate(names):
                key = _emb_key(str(filename))
                if key not in wanted:
                    continue
                if key in locations:
                    raise RuntimeError(f"duplicate source image key in image repo: {key}")
                locations[key] = (rgi, row_idx)
                hits.append((key, row_idx))
            if not hits:
                continue

            lo, hi = rg_span(md, rgi, columns=("image",))
            remote._span = (lo, _fetch_range(url, lo, hi - 1))
            image_col = pq.ParquetFile(remote).read_row_group(rgi, columns=["image"]).column(
                "image"
            )
            for key, row_idx in hits:
                payload = image_col[row_idx].as_py()
                if isinstance(payload, dict):
                    payload = payload.get("bytes")
                if not payload:
                    raise ValueError(f"image repo row has no image bytes for {key}")
                with Image.open(io.BytesIO(payload)) as image:
                    sizes[key] = image.size
            print(f"[val-grid] {shard} row-group {rgi}: fetched {len(hits)} selected images",
                  flush=True)
            if len(sizes) == len(wanted):
                print(f"[val-grid] found all {len(wanted)} selected images", flush=True)
                return sizes
        print(f"[val-grid] scanned image shard {shard_no}/{len(files)}; "
              f"found {len(sizes)}/{len(wanted)}", flush=True)
        if len(sizes) == len(wanted):
            return sizes

    missing = wanted - set(sizes)
    raise FileNotFoundError(
        f"image repo scan could not find {len(missing)} selected validation images; "
        f"first keys: {sorted(missing)[:5]}"
    )


def backfill_rows(rows, image_sizes, index):
    """Return copied rows with image-derived grids; fail on missing/mismatch."""
    import torch

    from vision_adapter.core import grid_from_dims, placeholder_count

    targets = {row["emb"] for row in rows if row.get("g") == "agentic" and row.get("emb")}
    missing_images = targets - set(image_sizes)
    if missing_images:
        raise FileNotFoundError(
            f"could not resolve {len(missing_images)} validation agentic images; "
            f"first keys: {sorted(missing_images)[:5]}"
        )

    grids = {}
    for key in sorted(targets):
        loc = index.get(key)
        if loc is None:
            raise KeyError(f"validation embedding is absent from the full key index: {key}")
        if len(loc) < 3:
            raise RuntimeError(f"key index has no stored n_vis to verify measured grid: {key}")
        grid = grid_from_dims(*image_sizes[key]).tolist()
        n_vis = placeholder_count(torch.as_tensor(grid), 2)
        if n_vis != int(loc[2]):
            raise ValueError(
                f"image-derived grid mismatch for {key}: grid={grid} -> {n_vis} tokens, "
                f"embedding index says {loc[2]}"
            )
        grids[key] = [int(v) for v in grid]

    out = []
    for row in rows:
        new = dict(row)
        if row.get("g") == "agentic" and row.get("emb"):
            measured = grids[row["emb"]]
            existing = row.get("grid_thw")
            if existing is not None and [int(v) for v in existing] != measured:
                raise ValueError(f"existing grid_thw disagrees with source image for {row['emb']}")
            new["grid_thw"] = measured
        out.append(new)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/train_manifest_val_disjoint.jsonl")
    ap.add_argument("--images-dir", default=None,
                    help="optional local root containing group dirs such as agentic/; "
                         "if omitted, selected image rows are range-fetched from HF")
    ap.add_argument("--images-repo", default="keypa/vision-adapter-images")
    ap.add_argument("--out", default="data/val_generation_measured.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-vis-tokens", type=int, default=300)
    ap.add_argument("--max-vis-tokens", type=int, default=380)
    ap.add_argument("--data-dir", default="data",
                    help="cache location for the full embedding key index")
    ap.add_argument("--train-manifest", default=None,
                    help="local train manifest; fetched measured train manifest if omitted")
    args = ap.parse_args()

    from vision_adapter.data.stream import build_key_index, fetch_manifest, list_shards
    from vision_adapter.manifest import load_manifest, write_manifest_with_header

    manifest_path = Path(args.manifest)
    images_dir = Path(args.images_dir) if args.images_dir else None
    if not manifest_path.is_file():
        raise FileNotFoundError(f"validation manifest not found: {manifest_path}")
    if images_dir is not None and not images_dir.is_dir():
        raise FileNotFoundError(f"local image directory not found: {images_dir}")
    rows, header = load_manifest(manifest_path)
    agentic_keys = [row["emb"] for row in rows if row.get("g") == "agentic" and row.get("emb")]
    if not agentic_keys:
        raise RuntimeError("validation manifest contains no agentic embedding keys")
    if args.n <= 0 or args.min_vis_tokens < 0 or args.max_vis_tokens < args.min_vis_tokens:
        ap.error("--n must be positive and token bounds must satisfy 0 <= min <= max")
    if len(set(agentic_keys)) != len(agentic_keys):
        raise RuntimeError("validation manifest has duplicate agentic embedding keys")

    if args.train_manifest:
        train_path = Path(args.train_manifest)
        if not train_path.is_file():
            raise FileNotFoundError(f"train manifest not found: {train_path}")
        train_rows, _ = load_manifest(train_path)
    else:
        train_rows = fetch_manifest(cache_dir=str(Path(args.data_dir) / "cache"))
    train_keys = {row.get("emb") for row in train_rows if row.get("emb")}
    overlap = set(agentic_keys) & train_keys
    if overlap:
        raise RuntimeError(f"validation agentic keys overlap training manifest: {len(overlap)}")
    print(f"[val-grid] {len(agentic_keys)} agentic keys; train overlap=0", flush=True)

    shard_order = list_shards()
    index = build_key_index(shard_order, cache_dir=str(Path(args.data_dir) / "cache"))
    chosen, counts = select_val_rows_by_n_vis(
        rows, index, n=args.n, seed=args.seed,
        min_n_vis=args.min_vis_tokens, max_n_vis=args.max_vis_tokens,
    )
    print(f"[val-grid] eligible/selected={counts['eligible']}/{counts['selected']} "
          f"in n_vis={args.min_vis_tokens}..{args.max_vis_tokens}; "
          f"missing index={counts['missing_index']}", flush=True)
    if not chosen:
        raise RuntimeError("no disjoint agentic validation rows in the requested n_vis range")
    if len(chosen) < args.n:
        print(f"[val-grid] only {len(chosen)} of {args.n} requested rows are available", flush=True)
    chosen_keys = {row["emb"] for row in chosen}
    if images_dir is not None:
        print(f"[val-grid] searching local originals for {len(chosen)} selected rows under {images_dir}", flush=True)
        image_paths = find_images(images_dir, chosen_keys)
        image_sizes = image_sizes_from_paths(image_paths)
    else:
        print(f"[val-grid] range-fetching {len(chosen)} selected originals from {args.images_repo}", flush=True)
        image_sizes = fetch_remote_image_sizes(chosen_keys, repo=args.images_repo)
    measured_rows = backfill_rows(chosen, image_sizes, index)
    grids_written = sum(1 for row in measured_rows
                        if row.get("g") == "agentic" and row.get("grid_thw") is not None)
    tags = dict(header.tags or {}) if header else {}
    tags.update({
            "geometry_source": "original local image dimensions + grid_from_dims",
            "geometry_verified_against_embedding_n_vis": True,
            "agentic_rows_with_grid": grids_written,
            "agentic_rows_selected": len(chosen),
            "selection_n_vis_min": args.min_vis_tokens,
            "selection_n_vis_max": args.max_vis_tokens,
            "selection_seed": args.seed,
        })
    out_path = write_manifest_with_header(
        args.out,
        measured_rows,
        seeds=header.seeds if header else None,
        upstream=header.upstream if header else None,
        tags=tags,
    )
    print(f"[val-grid] wrote {out_path}: measured grids {grids_written}/{len(agentic_keys)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

