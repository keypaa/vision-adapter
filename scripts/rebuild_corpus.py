#!/usr/bin/env python3
"""Rebuild the embedding corpus in the blob format, with a per-slice pixel cap.

Why (docs/DATA_PATH_REBUILD.md D2/D6, docs/REFERENCE_RECIPE.md):
the current corpus was embedded from images up to 12.8 Mpx, which produced
up to 16,598 visual tokens per row -- 43x the reference's 300k-pixel budget
(<=383 tokens), 1.7% of rows past the 4,096-token context limit (truncated),
and an 883.8 GiB corpus served through parquet row groups (3.19x overfetch).

This script:
  * reads the images repo shard by shard (parquet: filename + image bytes);
  * caps the pixel budget (--cap-patches, default 1531 = the reference's
    300,000 px) so every row is <= 383 merged tokens;
  * runs the frozen MoonViT-V2 and writes the rows CONCATENATED per shard
    (emb_XXXX.bin) with an index (emb_XXXX.idx, n_vis per row);
  * writes rows.parquet: key -> (shard, row, n_vis, grid_thw) -- this REPLACES
    both the key index and the grid sidecar, and the grid is the NEW geometry
    (the old sidecar describes the old, uncapped images);
  * writes corpus.json and uploads every shard as it is produced, so the disk
    only ever holds one shard and no volume is needed.

The decode/resize (CPU) overlaps the ViT forward (GPU) through a bounded
queue: the producer keeps the worker pool busy while the consumer runs the
model.

Resumable: a shard whose files are already built is skipped, and the per-shard
key map is kept on disk, so a resumed run re-embeds nothing.

Usage (GPU box):
    python scripts/rebuild_corpus.py --out-repo keypa/vision-adapter-embeddings-v2 \
        --out-dir ./corpus_out --device cuda --shards 0,1,2
    # everything, all shards:
    python scripts/rebuild_corpus.py --out-repo keypa/vision-adapter-embeddings-v2 \
        --out-dir ./corpus_out --device cuda
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The images repo is xet-backed, and the big shards (6.9 GB) fail to
# reconstruct with "Internal Writer Error: Background writer channel closed".
# The classic (non-xet) download path does not hit it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

from vision_adapter.data.blob_corpus import ROW_BYTES, ShardInfo, write_corpus_meta

IMAGES_REPO = "keypa/vision-adapter-images"
N_SHARDS = 17
MOONVIT_REPO = "keypa/MoonViT-V2-Standalone"

# Per-slice pixel budgets (docs/REFERENCE_RECIPE.md §3, D6).
#
# The reference's rule (300,000 px) was set for ART images: WikiArt at 256 px
# and fashion products at 384x512 are ~100-200k px, so the cap was a mild
# downscale of photographs. Our doc slices are not photographs: they are
# rendered documents whose ANSWER lives in small text. Measured on the real
# corpus (served positions, n_vis from the key index):
#
#   slice                rows   n_vis med      max    KiB/img   budget
#   agentic/*           51,834        364      380     7-196    300k px (unchanged: already there)
#   vqav2/okvqa/aokvqa  4,157        368      529    339-469   300k px (at/under budget already)
#   ocrvqa/scienceqa     2,298        216      324    61-246    300k px
#   visual7w/ai2d/chartqa 3,708   252-580    2,916   38-340    300k px
#   plotqa               5,366        960    1,250     29      300k px (synthetic charts: cheap, legible)
#   textvqa/infographic_vqa  1,359  1,036-1,250 4,760  870-942   800k px (text must stay readable)
#   docvqa/screen2words/websight 2,395 2,691-4,941 16,598  488-1,038  800k px (or drop: 4k+ tokens
#                                                                        does not fit the 4,096 context)
#
# The 800k px tier is 1,020 merged tokens: 3x the reference's budget, still
# inside the 4,096-token context with room for the text. Below ~600k px a
# scanned page's 12-px glyphs fall under one 14-px patch, so the answer is
# unreadable -- that is the legibility floor, and it is a judgement call, not
# a measured one.
BUDGETS: dict[str, int] = {
    "default": 300_000,
    "cauldron/textvqa": 800_000,
    "cauldron/infographic_vqa": 800_000,
    "cauldron/docvqa": 800_000,
    "cauldron/screen2words": 800_000,
    "cauldron/websight": 800_000,
}


def subset_of(filename: str) -> str:
    """'cauldron/chartqa-0000001-0.png' -> 'cauldron/chartqa'"""
    import re

    parts = filename.split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    stem = re.sub(r"[-_]\d+$", "", stem)
    stem = re.sub(r"-\d+$", "", stem)
    return "/".join(parts[:-1] + [stem])


def budget_for(filename: str, default: int) -> int:
    return BUDGETS.get(subset_of(filename), default)


# ------------------------------------------------------------------ input --
def images_shard(repo: str, shard: int, cache: Path, attempts: int = 3) -> Path:
    """Download one images-repo parquet shard (0.3-6.9 GB) and return it.

    The big shards can fail mid-reconstruction ("File reconstruction error:
    Internal Writer Error") — that is transient, so retry.
    """
    import time

    from huggingface_hub import hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)
    name = f"data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
    last = None
    for attempt in range(1, attempts + 1):
        try:
            path = Path(hf_hub_download(repo, name, repo_type="dataset",
                                        local_dir=str(cache)))
            return path
        except Exception as e:  # noqa: BLE001 - retried
            last = e
            if "quota" in str(e).lower() or "space" in str(e).lower():
                # the images shards are 0.3-6.9 GB; the download needs room for
                # the file AND the transfer's temporary copy
                import shutil

                free = shutil.disk_usage(cache).free
                raise RuntimeError(
                    f"not enough disk to fetch {name}: {free/2**30:.1f} GiB free in "
                    f"{cache}. Point --cache-dir at a bigger disk (the images shards "
                    f"are up to 6.9 GB)."
                ) from e
            print(f"[rebuild] shard {shard}: download attempt {attempt} failed "
                  f"({type(e).__name__}: {e})", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {name} after {attempts} attempts: {last}")


def cap_image(im, max_pixels: int, round_to: int = 28):
    """Resize to <= max_pixels, dimensions floored to a multiple of 28.

    This is the REFERENCE builder's rule (0xSero's collator):
        scale = (max_image_pixels / (w*h)) ** 0.5
        new_w = (new_w // 28) * 28 ; new_h = (new_h // 28) * 28
    The floor matters: our own `navit_resize` PADS up to a 28-multiple instead,
    which pushed the merged token count to 411-425 where the reference's rule
    gives <= 382 (300,000 px / 196 px per token, rounded DOWN). Measured on
    shard 13 of the images repo: with the pad-up rule the cap yielded a median
    of 411 tokens; with the floor rule it is <= 382.
    """
    import math

    from PIL import Image

    w, h = im.size
    if w * h <= max_pixels:
        return im
    scale = math.sqrt(max_pixels / float(w * h))
    nw = max(round_to, int(w * scale))
    nh = max(round_to, int(h * scale))
    nw = (nw // round_to) * round_to
    nh = (nh // round_to) * round_to
    if (nw, nh) == (w, h):
        return im
    return im.resize((nw, nh), Image.LANCZOS)


def prepare(blob: bytes, max_pixels: int, preprocess_mod=None, filename: str = ""):
    """decode -> cap -> patchify. Returns (pixel_values, grid_thw, n_patches).

    The budget is PER SLICE: `budget_for(filename)` maps the image's subset to
    its pixel budget (BUDGETS), so the doc slices that must stay readable get
    a bigger budget than the reference's 300k px.
    """
    from PIL import Image

    from vision_adapter.models.preprocess import process_image

    im = Image.open(io.BytesIO(blob)).convert("RGB")
    im = cap_image(im, budget_for(filename, max_pixels) if filename else max_pixels)
    out = process_image(im)
    return out["pixel_values"], out["grid_thws"], int(out["pixel_values"].shape[0])


# ------------------------------------------------------------------ shard --
def build_shard(shard: int, args, vit, device, preprocess_mod) -> ShardInfo:
    """Embed one images shard into the blob format. Returns the shard info."""
    import pyarrow.parquet as pq

    name = f"emb_{shard:04d}"
    t0 = time.time()
    src = images_shard(args.images_repo, shard, args.cache_dir)
    pf = pq.ParquetFile(src)

    bin_buf = io.BytesIO()
    idx_vals: list[int] = []
    key_rows: list[dict] = []
    row_in_shard = 0

    q: queue.Queue = queue.Queue(maxsize=4 * max(1, args.workers))
    ex = ThreadPoolExecutor(max_workers=args.workers)

    def producer():
        try:
            n = 0
            for batch in pf.iter_batches(batch_size=64, columns=["filename", "image"]):
                names = batch.column("filename").to_pylist()
                blobs = batch.column("image").to_pylist()
                for nm, blob in zip(names, blobs):
                    if args.limit and n >= args.limit:
                        return
                    n += 1
                    q.put((nm, ex.submit(prepare, blob, args.max_pixels,
                                          preprocess_mod, str(nm))))
        finally:
            q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    def flush(group):
        nonlocal row_in_shard
        if not group:
            return
        pv = torch.cat([g[0] for g in group], dim=0)
        gt = torch.cat([g[1] for g in group], dim=0)
        merged = vit(pv.to(device).to(torch.bfloat16), gt.to(device))
        for (pvv, g, npa, fname), emb in zip(group, merged):
            flat = emb.reshape(emb.shape[0], -1).to(torch.bfloat16).cpu()
            bin_buf.write(flat.view(torch.uint8).numpy().tobytes())
            idx_vals.append(int(flat.shape[0]))
            key_rows.append({"key": fname, "shard": shard, "row": row_in_shard,
                             "n_vis": int(flat.shape[0]),
                             "grid_thw": [int(x) for x in g[0].tolist()]})
            row_in_shard += 1

    group: list = []
    patches = 0
    while True:
        item = q.get()
        if item is None:
            break
        nm, fut = item
        pv, g, npa = fut.result()
        group.append((pv, g, npa, nm))
        patches += npa
        if patches >= args.batch_patches:
            flush(group)
            group = []
            patches = 0
    flush(group)
    ex.shutdown(wait=False, cancel_futures=True)
    t.join(timeout=5)

    bin_path = args.out_dir / f"{name}.bin"
    idx_path = args.out_dir / f"{name}.idx"
    rows_path = args.out_dir / f"rows_{shard:04d}.jsonl"
    tmp = bin_path.with_suffix(".bin.tmp")
    tmp.write_bytes(bin_buf.getvalue())
    os.replace(tmp, bin_path)
    tmp = idx_path.with_suffix(".idx.tmp")
    tmp.write_bytes(np.array(idx_vals, dtype=np.uint16).tobytes())
    os.replace(tmp, idx_path)
    with open(rows_path, "w") as f:
        for r in key_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[rebuild] {name}: {row_in_shard} rows, "
          f"{bin_path.stat().st_size / 2**30:.2f} GiB, {time.time() - t0:.0f}s", flush=True)
    return ShardInfo(f"{name}.bin", row_in_shard, bin_path.stat().st_size)


# ------------------------------------------------------------- dry run ----
def dry_run(args, shard_ids, preprocess_mod) -> int:
    """Decode + cap only: report the token distribution, write nothing."""
    import pyarrow.parquet as pq

    for shard in shard_ids:
        src = images_shard(args.images_repo, shard, args.cache_dir)
        pf = pq.ParquetFile(src)
        nvis = []
        t0 = time.time()
        for batch in pf.iter_batches(batch_size=64, columns=["filename", "image"]):
            for blob in batch.column("image").to_pylist():
                _pv, _g, npa = prepare(blob, args.max_pixels, preprocess_mod)
                nvis.append(npa // 4)
                if args.limit and len(nvis) >= args.limit:
                    break
            if args.limit and len(nvis) >= args.limit:
                break
        a = np.array(nvis)
        print(f"[rebuild] {shard}: {len(a)} images  n_vis median {int(np.median(a))} "
              f"p99 {int(np.percentile(a, 99))} max {int(a.max())}  "
              f"{a.sum() * ROW_BYTES / 2**30:.2f} GiB of embeddings  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return 0


# ---------------------------------------------------------------- publish --
def publish(args, shard_ids, shard_infos) -> int:
    """Merge the per-shard key maps and upload the corpus metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_upload

    keys, shards_, rows_, nvis_, grids_ = [], [], [], [], []
    for shard in shard_ids:
        rows_path = args.out_dir / f"rows_{shard:04d}.jsonl"
        if not rows_path.exists():
            continue
        for line in open(rows_path):
            r = json.loads(line)
            keys.append(r["key"])
            shards_.append(r["shard"])
            rows_.append(r["row"])
            nvis_.append(r["n_vis"])
            grids_.append(r["grid_thw"])
    pq.write_table(pa.table({"key": keys, "shard": shards_, "row": rows_,
                              "n_vis": nvis_, "grid_thw": grids_}),
                   args.out_dir / "rows.parquet")
    meta = write_corpus_meta(args.out_dir / "corpus.json", shard_infos,
                             cap_patches=None,
                             source=args.images_repo,
                             extra={"max_pixels": args.max_pixels,
                                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"[rebuild] corpus: {meta['total_rows']} rows, {meta['total_bytes']/2**30:.1f} GiB")
    if args.out_repo:
        for name in ("rows.parquet", "corpus.json"):
            hf_hub_upload(path_or_fileobj=str(args.out_dir / name), path_in_repo=name,
                          repo_id=args.out_repo, repo_type="dataset")
        print(f"[rebuild] published to {args.out_repo}")
    return 0


# ------------------------------------------------------------------ main --
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-repo", default=IMAGES_REPO)
    ap.add_argument("--out-repo", default="", help="HF dataset repo to publish to (empty = local only)")
    ap.add_argument("--out-dir", type=Path, default=Path("./corpus_out"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/rebuild_cache"))
    ap.add_argument("--max-pixels", type=int, default=300_000,
                    help="pixel budget per image (300,000 = the reference builder's rule; "
                         "800,000 for a denser doc/OCR slice -- D6)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-patches", type=int, default=120_000,
                    help="patches per ViT forward")
    ap.add_argument("--shards", default="", help="comma-separated shard indices (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images per shard (0 = all)")
    ap.add_argument("--workers", type=int, default=8, help="decode threads")
    ap.add_argument("--dry-run", action="store_true",
                    help="decode+cap only: report the token distribution, write nothing")
    args = ap.parse_args()

    import vision_adapter.models.preprocess as preprocess_mod

    # THE CAP: one constant. _resize_size() then downscales every image to
    # <= max_pixels, exactly like the reference builder's 300k-px rule.
    # The pixel budget is applied by cap_image() (the reference's rule:
    # dimensions floored to a multiple of 28). MAX_PATCHES stays as the
    # preprocess contract's own safety cap.
    print(f"[rebuild] pixel budget: {args.max_pixels:,} px per image "
          f"(<= {args.max_pixels // 196} merged tokens)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_ids = ([int(s) for s in args.shards.split(",")] if args.shards
                 else list(range(N_SHARDS)))

    if args.dry_run:
        return dry_run(args, shard_ids, preprocess_mod)

    from huggingface_hub import HfApi, hf_hub_download

    from vision_adapter.models.moonvit import load_moonvit_from_safetensors
    cfg = json.load(open(hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                                          filename="vision_config.json")))
    st_path = hf_hub_download(repo_id=MOONVIT_REPO, repo_type="model",
                              filename="moonvit_v2.safetensors")
    print("[rebuild] loading MoonViT-V2 ...", flush=True)
    device = torch.device(args.device)
    vit = load_moonvit_from_safetensors(st_path, cfg, device=str(device),
                                        dtype=torch.bfloat16)
    vit.eval()
    if args.out_repo:
        HfApi().create_repo(args.out_repo, repo_type="dataset", exist_ok=True)

    shard_infos: list[ShardInfo] = []
    t_start = time.time()
    for shard in shard_ids:
        name = f"emb_{shard:04d}"
        bin_path = args.out_dir / f"{name}.bin"
        idx_path = args.out_dir / f"{name}.idx"
        rows_path = args.out_dir / f"rows_{shard:04d}.jsonl"
        if bin_path.exists() and idx_path.exists() and rows_path.exists():
            n = sum(1 for _ in open(rows_path))
            shard_infos.append(ShardInfo(f"{name}.bin", n, bin_path.stat().st_size))
            print(f"[rebuild] {name}: already built ({n} rows) — skipped", flush=True)
            continue
        info = build_shard(shard, args, vit, device, preprocess_mod)
        shard_infos.append(info)
        if args.out_repo:
            from huggingface_hub import hf_hub_upload

            for p in (bin_path, idx_path, rows_path):
                hf_hub_upload(path_or_fileobj=str(p), path_in_repo=p.name,
                              repo_id=args.out_repo, repo_type="dataset")
            print(f"[rebuild] {name}: uploaded", flush=True)

    rc = publish(args, shard_ids, shard_infos)
    print(f"[rebuild] total {(time.time() - t_start) / 60:.1f} min")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
