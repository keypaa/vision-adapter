#!/usr/bin/env python3
"""Transfer bench: hf_transfer whole-shard vs Range fetch (NEXT_STEPS §3/Phase 5).

Decision-grade (not paper-grade): hf_transfer timing includes disk write,
Range timing is pure network (chunks discarded). Both fresh-cache.

Usage (Molab):
    python scripts/bench_transfer.py --shard data/emb_0025.parquet --data-dir ./data
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def time_download(method, fetch):
    """Time a zero-arg download callable; fetch returns byte count."""
    t0 = time.perf_counter()
    nbytes = fetch()
    dt = time.perf_counter() - t0
    return {"method": method, "bytes": int(nbytes),
            "seconds": round(dt, 3), "mib_s": round((nbytes / 2**20) / dt, 2) if dt > 0 else 0.0}


def _merge_record(path: Path, rec: dict) -> list:
    """Append/replace one method record in the bench JSON (multi-process runs)."""
    try:
        current = json.loads(path.read_text())
        recs = current.get("results", [])
    except Exception:
        recs = []
    recs = [r for r in recs if r.get("method") != rec.get("method")] + [rec]
    path.write_text(json.dumps({"results": recs}, indent=2))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/emb_0025.parquet")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--repo", default="keypa/vision-adapter-embeddings")
    ap.add_argument("--method", choices=("hf_transfer", "range"), default="range",
                    help="hf_transfer: whole-file hf_hub_download (needs "
                    "HF_HUB_ENABLE_HF_TRANSFER=1 in env, read at import); "
                    "range: chunked _fetch_range (repo retry logic). "
                    "NOTE: repo _download_shard_hf_transfer is Modal-gated, "
                    "so Molab compares the download backends directly.")
    args = ap.parse_args()

    from vision_adapter.data.stream import (
        RemoteShard,
        _fetch_range,
        _remote_size,
    )

    FETCH_CHUNK = RemoteShard.FETCH_CHUNK

    url = f"https://huggingface.co/datasets/{args.repo}/resolve/main/{args.shard}"
    size = _remote_size(url)
    print(f"[bench] {args.shard} {size / 2**30:.2f}GiB via {args.method}", flush=True)

    if args.method == "hf_transfer":
        def do_hf():
            import shutil
            import tempfile

            from huggingface_hub import hf_hub_download

            d = tempfile.mkdtemp(prefix="bench_hf_")
            try:
                p = hf_hub_download(args.repo, args.shard, repo_type="dataset", local_dir=d)
                return Path(p).stat().st_size
            finally:
                shutil.rmtree(d, ignore_errors=True)

        rec = time_download("hf_transfer", do_hf)
    else:
        def do_range():
            got = 0
            for lo in range(0, size, FETCH_CHUNK):
                got += len(_fetch_range(url, lo, min(lo + FETCH_CHUNK, size) - 1))
            return got

        rec = time_download("range", do_range)
    print(f"[bench] {rec['method']}: {rec['bytes'] / 2**30:.2f}GiB in {rec['seconds']}s = {rec['mib_s']}MiB/s",
          flush=True)
    out = Path(args.data_dir) / "bench_transfer.json"
    _merge_record(out, rec)
    print(f"[bench] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
