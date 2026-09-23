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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/emb_0025.parquet")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--repo", default="keypa/vision-adapter-embeddings")
    args = ap.parse_args()

    from vision_adapter.data.stream import (
        FETCH_CHUNK,
        _download_shard_hf_transfer,
        _fetch_range,
        _remote_size,
    )

    url = f"https://huggingface.co/datasets/{args.repo}/resolve/main/{args.shard}"
    size = _remote_size(url)
    print(f"[bench] {args.shard} {size / 2**30:.2f}GiB", flush=True)

    def do_hf():
        import shutil
        import tempfile

        d = tempfile.mkdtemp(prefix="bench_hf_")
        try:
            p = _download_shard_hf_transfer(args.shard, d)
            return Path(p).stat().st_size
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def do_range():
        got = 0
        for lo in range(0, size, FETCH_CHUNK):
            got += len(_fetch_range(url, lo, min(lo + FETCH_CHUNK, size) - 1))
        return got

    recs = [time_download("hf_transfer", do_hf), time_download("range", do_range)]
    for r in recs:
        print(f"[bench] {r['method']}: {r['bytes'] / 2**30:.2f}GiB in {r['seconds']}s = {r['mib_s']}MiB/s",
              flush=True)
    out = Path(args.data_dir) / "bench_transfer.json"
    out.write_text(json.dumps({"shard": args.shard, "results": recs}, indent=2))
    print(f"[bench] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
