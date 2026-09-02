#!/usr/bin/env python3
"""
scripts/local_repack_bucketed.py — Local bucketed repack, capped for 6-core / 12GiB laptop.

Does the same 930GiB bucketed rewrite as `modal run vision_adapter/data/pack.py::pack_bucketed`
but from this laptop, staying light:

- RAM: 12GiB cap (absolute max 14GiB) — batch_size 32, workers 4, one shard staged at a time (~9GiB peak)
- Disk: ~20GiB stage_dir (/tmp/emb_stage_local), per-shard cleanup, no 930GiB hold
- CPU: 4 workers (of 6 cores) so laptop stays usable for 4-5h
- Same result: 103 shards n_vis-homogeneous (6 buckets), overwrites keypa/vision-adapter-embeddings

Usage:
  HF_TOKEN=hf_xxx python scripts/local_repack_bucketed.py --hf-only  # writes to HF
  # or dry-run (no push): python scripts/local_repack_bucketed.py --dry-run

Requires: pip install -e .[train]  (torch, pyarrow, huggingface_hub, hf_transfer)
Handles HF token via --hf-token / $HF_TOKEN / Colab userdata, no hard fail if absent (anonymous for read, but push needs write).
"""

from __future__ import annotations

import argparse
import os
import time

# Cap CPU: don't oversubscribe 6 cores
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

SHARD_ROWS = 1360
VOL_NAME = "vision-adapter-data"
EMB_REPO = "keypa/vision-adapter-embeddings"


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(description="Local bucketed repack (capped 12GiB/4 workers/20GiB disk)")
    ap.add_argument("--shard-rows", type=int, default=SHARD_ROWS)
    ap.add_argument("--workers", type=int, default=4, help="download workers (cap 4 of 6 cores)")
    ap.add_argument("--batch-size", type=int, default=32, help="pack batch_size (cap RAM, default 64→32)")
    ap.add_argument("--stage-dir", default="/tmp/emb_stage_local", help="disk staging (~9GiB peak per shard)")
    ap.add_argument("--em-repo", default=EMB_REPO)
    ap.add_argument("--hf-only", action="store_true", help="push to HF only, skip /data/shards volume copy")
    ap.add_argument("--dry-run", action="store_true", help="compute n_vis sort only, no pack/push")
    ap.add_argument("--hf-token", default=None, help="HF write token (or HF_TOKEN env)")
    ap.add_argument("--only", default="", help="shard range i[:j] for resume, e.g. 0:10")
    args = ap.parse_args(argv)

    # HF token: CLI > env > Colab userdata, graceful if absent (read ok, push needs write)
    tok = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from vision_adapter.backends.auth import get_hf_token

            tok = get_hf_token()
        except Exception:
            pass
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        print(f"[local-repack] HF token present ({len(tok)} chars)", flush=True)
    else:
        print("[local-repack] HF token absent — read will be anonymous, push will fail (need write token)", flush=True)
        if not args.dry_run and args.hf_only:
            print("[local-repack] --hf-only without write token will fail at push — run with HF_TOKEN=hf_xxx", flush=True)

    # Only need modal for Volume source when doing real repack (not HF-only read)
    try:
        import modal

        vol = modal.Volume.from_name(VOL_NAME)
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
        print(f"[local-repack] Volume {VOL_NAME} + HF {EMB_REPO} ready", flush=True)
    except Exception as e:
        print(f"[local-repack] Modal not available ({e}) — will try HF Range fallback for n_vis (slower)", flush=True)
        vol = None
        api = None

    # Import capped pack helpers
    from vision_adapter.data.pack import sorted_embedding_names

    # If vol available, list via Volume (fast, no 930GiB download yet)
    if vol is not None:
        entries = vol.listdir("embeddings")
        names = sorted_embedding_names(entries)
        sizes = {e.path: e.size for e in entries}
    else:
        # Fallback: list via HF (slower, but works without Volume)
        from huggingface_hub import HfApi as _HfApi

        hf_api = _HfApi(token=tok)
        files = hf_api.list_repo_files(EMB_REPO, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("data/emb_") and f.endswith(".parquet"))
        # For fallback we can't do bucketed repack without .pt — abort
        print("[local-repack] HF fallback listing: bucketed repack needs Volume .pt source — aborting, use Modal", flush=True)
        return 2

    print(f"[local-repack] embeddings: {len(names)} shards: {(len(names)+args.shard_rows-1)//args.shard_rows} workers={args.workers} batch={args.batch_size} stage={args.stage_dir}", flush=True)

    # --- Bucketed sort (capped, in-memory n_vis map, no tmpdir) ---
    print("[local-repack] --bucketed: computing n_vis in-memory (capped 4 workers, ~3h on 6 cores) ...", flush=True)
    from collections import Counter
    import io
    import threading
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    nvis_map: dict[str, int] = {}
    t0 = time.time()
    last_log = [t0]
    total = len(names)
    stop_hb = threading.Event()

    def _bar(done, tot, w=40):
        f = int(w * done / max(1, tot))
        return "█" * f + "─" * (w - f)

    def _hb():
        while not stop_hb.wait(10):
            d = len(nvis_map)
            e = time.time() - t0
            r = d / max(1e-9, e)
            eta = (total - d) / max(1e-9, r) / 60 if r else 0
            print(f"[local-repack] heartbeat |{_bar(d,total)}| {d}/{total} ({100*d/total:.0f}%) {r:.0f} files/s ETA {eta:.0f}min", flush=True)

    hb = threading.Thread(target=_hb, daemon=True)
    hb.start()

    def _fetch(nm):
        try:
            buf = io.BytesIO()
            vol.read_file_into_fileobj(nm, buf)
            buf.seek(0)
            t = torch.load(buf, map_location="cpu", weights_only=True)
            return nm, int(t.shape[0])
        except Exception:
            return nm, 500

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_fetch, nm): nm for nm in names}
            for fut in as_completed(futs):
                nm, nv = fut.result()
                nvis_map[nm] = nv
                now = time.time()
                if now - last_log[0] >= 5 or len(nvis_map) == total:
                    last_log[0] = now
                    e = now - t0
                    r = len(nvis_map) / max(1e-9, e)
                    eta = (total - len(nvis_map)) / max(1e-9, r) / 60 if r else 0
                    print(f"[local-repack] n_vis |{_bar(len(nvis_map),total)}| {len(nvis_map)}/{total} ({100*len(nvis_map)/total:.0f}%) {r:.0f} files/s ETA {eta:.0f}min", flush=True)
    finally:
        stop_hb.set()
        hb.join(timeout=2)

    from vision_adapter.data.pack import _bucket_id

    scored = [(_bucket_id(nvis_map.get(nm, 500)), nm) for nm in names]
    scored.sort(key=lambda kv: (kv[0], kv[1]))
    bucketed_names = [nm for _, nm in scored]
    hist = Counter(_bucket_id(nvis_map.get(nm, 500)) for nm in names)
    print(f"[local-repack] histogram 0-100:{hist[0]} 101-500:{hist[1]} 501-1000:{hist[2]} 1001-2000:{hist[3]} 2001-4900:{hist[4]} 4901+:{hist[5]}", flush=True)
    print(f"[local-repack] bucketed order ready in {(time.time()-t0)/60:.1f}min", flush=True)

    if args.dry_run:
        print("[local-repack] --dry-run: stopping before pack/push (verify histogram above)", flush=True)
        return 0

    # --- Pack pipeline (capped) ---
    from vision_adapter.data.pack import run_pipeline

    names = bucketed_names
    # HF token required for push
    if not tok:
        print("[local-repack] no HF write token — aborting before push (set HF_TOKEN)", flush=True)
        return 2

    # Ensure stage_dir on disk (not tmpfs) and cap
    os.makedirs(args.stage_dir, exist_ok=True)
    lo, hi = 0, (len(names) + args.shard_rows - 1) // args.shard_rows
    if args.only:
        parts = args.only.split(":")
        lo = int(parts[0]) if parts[0] else 0
        hi = int(parts[1]) if len(parts) > 1 and parts[1] else hi

    print(f"[local-repack] packing shards {lo}:{hi} with workers={args.workers} batch={args.batch_size} hf_only={args.hf_only} (RAM 12GiB cap, disk ~9GiB/shard) ...", flush=True)
    # run_pipeline is already RAM-bounded (batch 32) and per-shard staged
    actions = run_pipeline(
        vol,
        api,
        names,
        args.shard_rows,
        args.stage_dir,
        args.em_repo,
        workers=args.workers,
        batch_size=args.batch_size,
        lo=lo,
        hi=hi,
        hf_only=args.hf_only,
        sizes=sizes,
        bucketed=True,
    )
    print(f"[local-repack] done actions {Counter(actions)} — verify HF 103 shards then delete volume", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
