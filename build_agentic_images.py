#!/usr/bin/env python3
"""
build_agentic_images.py — reconstruct referenced screenshot/UI images for the
0xSero/glm-vision-sft-mix dataset by positional join into upstream sources.

Join rule (verified): the trailing NN-number in a Sero filename is the 0-based
positional index into the upstream record list:
  waveui_NNNNNN.png      -> agentsea/wave-ui-25k      row N (parquet, embedded PNG bytes)
  showui_NNNNNN.png      -> showlab/ShowUI-desktop    row N (parquet, embedded PNG bytes)
  aitw_NNNNNN.jpg        -> xlangai/aguvis-stage2 aitw-l1.json[N]['image'] in aitw.zip
  guiact-web-multi_N..   -> aguvis guiact-web-multi (zip)
  mind2web_NNNNNN.jpg    -> aguvis mind2web (zip)
  miniwob_NNNNNN.jpg     -> aguvis miniwob (zip)

Outputs are resized (≤300k px, then floored to 28px multiples, min 28) and saved
to --out (default /tmp/opencode/agentic_images). Idempotent: existing files are
not rewritten.
"""

import argparse
import io
import json
import math
import os
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import requests
from PIL import Image
from huggingface_hub import hf_hub_download

MANIFEST = "/tmp/opencode/sero_manifest.parquet"
AGUVIS_PEEK_DIR = "/tmp/opencode/aguvis_peek"
AGUVIS_ZIP_DIR = "/tmp/opencode/aguvis_zips"
DEFAULT_OUT = "/tmp/opencode/agentic_images"

PARQUET_DATASETS = {
    "waveui": "agentsea/wave-ui-25k",
    "showui": "showlab/ShowUI-desktop",
}
AGUVIS_SUBSETS = ["aitw", "guiact-web-multi", "mind2web", "miniwob"]
ALL_SUBSETS = ["waveui", "showui", "aitw", "miniwob", "mind2web", "guiact-web-multi"]

FILENAME_RE = re.compile(r"^(waveui|showui|aitw|guiact-web-multi|mind2web|miniwob)_(\d+)\.(png|jpg)$")

WORKERS = 8
PROGRESS_EVERY = 500
USER_AGENT = "build-agentic-images/1.0"


# ---------------------------------------------------------------- manifest ---

def load_needed_indices(limit=None):
    """Return {subset: {pos_index: original_basename}} preserving exact filenames."""
    con = duckdb.connect()
    rows = con.execute(f"SELECT image FROM '{MANIFEST}'").fetchall()
    needed = {s: {} for s in ALL_SUBSETS}
    unmatched = []
    for (img,) in rows:
        m = FILENAME_RE.match(img)
        if not m:
            unmatched.append(img)
            continue
        needed[m.group(1)][int(m.group(2))] = img   # preserve exact original basename
    if unmatched:
        print(f"[warn] {len(unmatched)} manifest rows did not match any known prefix "
              f"(first: {unmatched[0]}) — skipped, not guessed.", file=sys.stderr)
    out = {}
    for s in ALL_SUBSETS:
        items = sorted(needed[s].items())           # [(idx, basename), ...]
        if limit is not None:
            items = items[:limit]
        out[s] = dict(items)
    return out


def sero_basename(subset, idx, ext):
    return f"{subset}_{idx:06d}.{ext}"


# -------------------------------------------------------- upstream (parquet) ---

def get_parquet_shard_urls(dataset):
    r = requests.get(
        f"https://datasets-server.huggingface.co/parquet?dataset={dataset}",
        timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return [f["url"] for f in r.json()["parquet_files"]]


def _url_to_relpath(url):
    """datasets-server url -> repo-relative data/... path the resolve endpoint serves."""
    m = re.search(r"/resolve/refs%2Fconvert%2Fparquet/(.+)$", url)
    return m.group(1) if m else url


def local_parquet_shards(dataset, needed_max_idx, counts=None):
    """Resume-safe, CDN-cached download of the shards covering needed_max_idx.

    `counts` is ignored on purpose (kept for call-site compatibility); we always
    ask HF for the full shard path set and let hf_hub_download's cache do the work.
    """
    if counts is None:
        counts = [None]  # placeholder; hf_hub_download does not need row counts
    n = shards_covering(counts, needed_max_idx) if counts[0] is not None else None
    urls = get_parquet_shard_urls(dataset)
    urls = urls[:n] if n else urls
    paths = []
    for i, u in enumerate(urls):
        rel = _url_to_relpath(u)
        # idempotent: re-uses the global HF hub cache
        p = hf_hub_download(repo_id=dataset, repo_type="dataset",
                            filename=rel, revision="refs/convert/parquet")
        paths.append(p)
        print(f"  [dl] {dataset} shard {i+1}/{len(urls)} -> {os.path.basename(p)}")
    return paths


def parquet_shard_row_counts(urls):
    counts = []
    for u in urls:
        df = duckdb.execute(
            f"SELECT num_rows FROM parquet_file_metadata('{u}')").fetchdf()
        counts.append(int(df["num_rows"][0]))
    return counts


def shards_covering(counts, upto_idx):
    """Return number of leading shards needed to cover global row index upto_idx."""
    acc = 0
    for i, c in enumerate(counts):
        acc += c
        if upto_idx < acc:
            return i + 1
    return len(counts)


def fetch_range_from_parquet(local_paths, counts, needed_indices):
    """
    Fetch {global_idx: image_bytes} for the given positional indices from the
    concatenation of LOCAL parquet shards (in order), via pyarrow streaming.
    """
    import pyarrow.parquet as pq
    want = set(needed_indices)
    out = {}
    offset = 0
    for path, n in zip(local_paths, counts):
        local = sorted(i - offset for i in want if offset <= i < offset + n)
        if local:
            need = set(local)
            tbl_iter = pq.ParquetFile(path).iter_batches(batch_size=2048,
                                                         columns=["image"])
            row_in_shard = 0
            for batch in tbl_iter:
                im = batch.column(0)
                for j in range(batch.num_rows):
                    if row_in_shard in need:
                        v = im[j].as_py()
                        b = v.get("bytes") if isinstance(v, dict) else v
                        if b is not None:
                            out[offset + row_in_shard] = bytes(b)
                    row_in_shard += 1
            print(f"  [parquet] {os.path.basename(path)}: +{len(local)} rows "
                  f"(global {offset + local[0]}..{offset + local[-1]})")
        offset += n
        if offset > max(want):
            break
    return out


# ---------------------------------------------------------- upstream (aguvis) ---

def aguvis_manifest_path(name):
    return os.path.join(AGUVIS_PEEK_DIR, f"{name}-l1.json")


def load_aguvis_manifest(name):
    """Load the per-subset manifest. Falls back to fetching from HF Hub if missing."""
    path = aguvis_manifest_path(name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"  [aguvis] {name}-l1.json not on disk; fetching from HF hub ...")
        fetched = hf_hub_download(
            repo_id="xlangai/aguvis-stage2",
            repo_type="dataset",
            filename=f"{name}-l1.json",
            local_files_only=False,
        )
        # Copy into the expected local path so reruns are instant and idempotent
        import shutil
        shutil.copyfile(fetched, path)
    with open(path) as f:
        return json.load(f)


def aguvis_zip_path(name):
    return os.path.join(AGUVIS_ZIP_DIR, f"{name}.zip")


def download_aguvis_zip(name):
    """Download <name>.zip via HF's parallel chunked downloader (fast, CDN-aware).

    hf_hub_download uses multiple connections + CloudFront CDN vs the
    single-connection requests.get which peaks at ~3 MB/s for these zips."""
    from huggingface_hub import hf_hub_download
    os.makedirs(AGUVIS_ZIP_DIR, exist_ok=True)
    dest = aguvis_zip_path(name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [zip] {name}.zip already complete ({os.path.getsize(dest)} bytes)")
        return dest
    print(f"  [zip] downloading {name}.zip via hf_hub_download ...")
    downloaded = hf_hub_download(
        repo_id="xlangai/aguvis-stage2",
        repo_type="dataset",
        filename=f"{name}.zip",
        local_dir=AGUVIS_ZIP_DIR,
        local_dir_use_symlinks=False,
    )
    # hf_hub_download saves to {local_dir}/xlangai--aguvis-stage2/{name}.zip
    if downloaded != dest and os.path.exists(downloaded):
        os.rename(downloaded, dest)
        # clean up HF cache dir
        hf_cache_dir = os.path.dirname(downloaded)
        if os.path.isdir(hf_cache_dir) and not os.listdir(hf_cache_dir):
            os.rmdir(hf_cache_dir)
    print(f"  [zip] {name}.zip ready ({os.path.getsize(dest)} bytes)")
    return dest


def fetch_from_aguvis_zip(name, needed_indices, manifest):
    """Return {global_idx: image_bytes} by looking up original filenames in the zip."""
    zip_path = download_aguvis_zip(name)
    want = {}
    for i in needed_indices:
        if i >= len(manifest):
            raise IndexError(f"{name}: index {i} >= manifest length {len(manifest)} — "
                             f"positional join not confirmed; refusing to guess")
        want[i] = manifest[i]["image"]
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        basenames = {}
        for n in names:
            basenames.setdefault(os.path.basename(n), n)
        for i, orig in want.items():
            member = orig if orig in names else basenames.get(os.path.basename(orig))
            if member is None:
                raise KeyError(f"{name}: '{orig}' (index {i}) not found in {name}.zip — "
                               f"positional join not confirmed; refusing to guess")
            out[i] = zf.read(member)
    print(f"  [zip] {name}: extracted {len(out)} images")
    return out


# ------------------------------------------------------------------ imaging ---

def resize_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = math.sqrt(300000 / (w * h))
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.LANCZOS)
    w, h = img.size
    tw, th = max(28, (w // 28) * 28), max(28, (h // 28) * 28)
    if (tw, th) != (w, h):
        img = img.resize((tw, th), Image.LANCZOS)
    return img


def process_one(args):
    """Decode, resize, save. Returns (basename, out_path, nbytes) or raises."""
    basename, img_bytes, ext, out_dir = args
    out_path = os.path.join(out_dir, basename)
    img = Image.open(io.BytesIO(img_bytes))
    img.load()
    img = resize_image(img)
    if ext == "png":
        if img.mode not in ("RGB", "RGBA", "L", "P"):
            img = img.convert("RGB")
        img.save(out_path + ".tmp", format="PNG")
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path + ".tmp", format="JPEG", quality=90)
    os.replace(out_path + ".tmp", out_path)
    return basename, out_path, os.path.getsize(out_path)


def output_exists(out_dir, subset, idx, ext):
    """Idempotency check: file exists and decodes (correct size = correctly produced)."""
    p = os.path.join(out_dir, sero_basename(subset, idx, ext))
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return False
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------- dry-run ---

def dry_run(needed):
    print("=" * 78)
    print("DRY-RUN: positional coverage table (NO zip/parquet bytes touched)")
    print("=" * 78)
    hdr = f"{'subset':<18} {'needed':>7} {'min':>7} {'max':>7} {'upstream':>34} {'up_size':>8} {'join':>6}"
    print(hdr)
    print("-" * len(hdr))
    parquet_meta = {}
    for subset in ALL_SUBSETS:
        items = needed[subset]
        idxs = sorted(items.keys())
        if not idxs:
            print(f"{subset:<18} {0:>7} {'-':>7} {'-':>7} {'-':>34} {'-':>8} {'SKIP':>6}")
            continue
        mn, mx = idxs[0], idxs[-1]
        if subset in PARQUET_DATASETS:
            ds = PARQUET_DATASETS[subset]
            urls = get_parquet_shard_urls(ds)
            counts = parquet_shard_row_counts(urls)
            parquet_meta[subset] = (urls, counts)
            up = ds + f" ({len(urls)} shards)"
            size = sum(counts)
        else:
            manifest = load_aguvis_manifest(subset)
            up = f"aguvis {subset}-l1.json + {subset}.zip"
            size = len(manifest)
        ok = "OK" if mx < size else "FAIL"
        print(f"{subset:<18} {len(idxs):>7} {mn:>7} {mx:>7} {up:>34} {size:>8} {ok:>6}")
    print("-" * len(hdr))
    print("Legend: needed=#distinct Sero indices; min/max=positional index range;")
    print("        up_size=len(upstream record list); join OK iff max < up_size.")
    return parquet_meta


# ---------------------------------------------------------------------- main ---

def build_subset(subset, idx_map, dry_meta=None):
    """idx_map: {pos_index: original_basename}. Returns {pos_index: image_bytes}."""
    todo = sorted(idx_map.keys())
    if not todo:
        print(f"[{subset}] nothing to do")
        return {}
    if subset in PARQUET_DATASETS:
        if dry_meta and subset in dry_meta:
            urls, counts = dry_meta[subset]
        else:
            urls = get_parquet_shard_urls(PARQUET_DATASETS[subset])
            counts = parquet_shard_row_counts(urls)
        mx = todo[-1]
        if mx >= sum(counts):
            raise IndexError(f"{subset}: index {mx} >= upstream size {sum(counts)} — "
                             f"positional join not confirmed; refusing to guess")
        nshards = shards_covering(counts, mx)
        print(f"[{subset}] need {len(todo)} images, max idx {mx} -> first {nshards}/{len(urls)} shards")
        paths = local_parquet_shards(PARQUET_DATASETS[subset], mx, counts)
        data = fetch_range_from_parquet(paths, counts[:nshards], todo)
    else:
        manifest = load_aguvis_manifest(subset)
        mx = todo[-1]
        if mx >= len(manifest):
            raise IndexError(f"{subset}: index {mx} >= manifest length {len(manifest)} — "
                             f"positional join not confirmed; refusing to guess")
        data = fetch_from_aguvis_zip(subset, todo, manifest)
    missing = [i for i in todo if i not in data]
    if missing:
        raise RuntimeError(f"{subset}: failed to resolve {len(missing)} indices "
                           f"(first: {missing[0]})")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=ALL_SUBSETS + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N needed indices per subset")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve indices + verify counts per subset, touching NO "
                         "zip/parquet bytes; print positional coverage table")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    needed = load_needed_indices(limit=args.limit)
    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]

    if args.dry_run:
        dry_run(needed)
        return 0

    os.makedirs(args.out, exist_ok=True)
    total_written = 0
    total_skipped = 0
    processed = 0
    for subset in subsets:
        idx_map = needed[subset]                      # {idx: basename}
        pending_map = {i: b for i, b in idx_map.items()
                       if not output_exists(args.out, subset, i, idx_map[i].rsplit('.', 1)[-1])}
        skipped = len(idx_map) - len(pending_map)
        total_skipped += skipped
        print(f"[{subset}] {len(idx_map)} needed, {skipped} already present, "
              f"{len(pending_map)} to fetch")
        if not pending_map:
            continue
        data = build_subset(subset, pending_map)      # {idx: image_bytes}
        jobs = [(idx_map[i], data[i], idx_map[i].rsplit('.', 1)[-1], args.out)
                for i in sorted(pending_map)]
        written = 0
        with ThreadPoolExecutor(max_workers=min(args.workers, WORKERS)) as ex:
            futs = {ex.submit(process_one, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                basename, out_path, nbytes = fut.result()
                written += 1
                processed += 1
                if processed % PROGRESS_EVERY == 0:
                    print(f"[progress] {processed} images written (latest: {basename}, "
                          f"{nbytes} bytes)")
        total_written += written
        print(f"[{subset}] wrote {written} images to {args.out}")
    print(f"DONE: wrote {total_written}, skipped {total_skipped} (already present), "
          f"out dir: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
