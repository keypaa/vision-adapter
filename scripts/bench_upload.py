"""Measure the HF upload rate (the unmeasured cost of publishing a new corpus).

The download side is known (311-349 MB/s from the rental box, 22-28 MB/s from
a laptop). The upload side decides whether a 344 GiB corpus can be published
from a GPU box in minutes or hours -- and nothing in the repo measures it.

Uploads a throwaway file to a scratch repo, times it, then deletes the repo.

Usage:
    python scripts/bench_upload.py --size-mib 1024
"""
import argparse
import os
import shutil
import tempfile
import time

REPO = "keypa/_upload_bench"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mib", type=int, default=1024)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, upload_file

    api = HfApi()
    tmp = tempfile.mkdtemp(prefix="upload_bench_")
    path = os.path.join(tmp, "blob.bin")
    chunk = os.urandom(1 << 20)
    with open(path, "wb") as f:
        for _ in range(args.size_mib):
            f.write(chunk)
    size = os.path.getsize(path)
    print(f"[bench] uploading {size/2**20:.0f} MiB to {args.repo} ...")
    t0 = time.perf_counter()
    try:
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        upload_file(
            path_or_fileobj=path, path_in_repo="blob.bin", repo_id=args.repo,
            repo_type="dataset",
        )
    except Exception as e:
        print(f"[bench] upload failed (token may be read-only): {type(e).__name__}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return 1
    dt = time.perf_counter() - t0
    print(f"[bench] uploaded in {dt:.1f}s -> {size/2**20/dt:.1f} MiB/s "
          f"({size/2**30/dt:.2f} GiB/s)")
    print(f"[bench] projection: 344 GiB in {344/(size/2**30/dt)/3600:.2f} h")
    if not args.keep:
        try:
            api.delete_repo(args.repo, repo_type="dataset")
            print(f"[bench] deleted scratch repo {args.repo}")
        except Exception as e:
            print(f"[bench] could not delete {args.repo}: {e}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
