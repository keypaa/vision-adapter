from typing import Iterator, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import time
from concurrent.futures import ThreadPoolExecutor

import os

SHARD_ROWS = 1360
VOL_NAME = "vision-adapter-data"
EMB_REPO = "keypa/vision-adapter-embeddings"
REPO_TYPE = "dataset"


class FileEntryLike:
    """Minimal FileEntry shim for tests (only `path` is needed)."""

    __slots__ = ("path",)

    def __init__(self, path):
        self.path = path


def sorted_embedding_names(entries):
    """Return sorted `embeddings/<sha1>.pt` paths (matches Modal's sorted(glob)).

    Tolerates both FileEntry.path forms: volume-root-relative
    (`embeddings/<name>.pt`) and directory-relative (`<name>.pt`)."""
    out = []
    for e in entries:
        p = e.path
        if p.startswith("embeddings/"):
            out.append(p)
        elif "/" in p:
            continue  # entry from another directory — not an embedding
        else:
            out.append(f"embeddings/{p}")
    return sorted(out)


def shard_slices(names, shard_rows):
    """Contiguous slices of `names` aligned to Modal's shard numbering."""
    out = []
    for i in range(0, len(names), shard_rows):
        out.append(names[i : i + shard_rows])
    return out


SCHEMA = pa.schema(
    [
        pa.field("key", pa.string()),
        pa.field("n_vis", pa.int64()),
        pa.field("vis_bytes", pa.binary()),
    ]
)


def make_row(path: str, tensor: torch.Tensor) -> dict:
    assert tensor.dim() == 2 and tensor.shape[-1] == 4096, path
    return {
        "key": path,
        "n_vis": int(tensor.shape[0]),
        "vis_bytes": tensor.view(torch.uint8).numpy().tobytes(),
    }


def iter_rows(local_paths: list[str]) -> Iterator[dict]:
    """torch.load each staged .pt, yield a row dict (key = embeddings/<basename>)."""
    for p in local_paths:
        t = torch.load(p, map_location="cpu", weights_only=True)
        yield make_row(f"embeddings/{os.path.basename(p)}", t)


def pack_rows(rows: Iterable[dict], out_path: str, batch_size: int = 64) -> None:
    """Stream rows to a parquet file in fixed-size batches (RAM-bounded).

    compression=None: bf16 float payloads are incompressible — snappy only
    burns CPU (measured 2.3x write time for ~0% size change)."""
    writer = pq.ParquetWriter(out_path, SCHEMA, compression=None)
    batch = []
    for r in rows:
        batch.append(r)
        if len(batch) >= batch_size:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            batch.clear()
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    writer.close()


def existing_volume_shards(vol):
    out = set()
    try:
        for e in vol.listdir("shards"):
            if e.path.endswith(".parquet"):
                out.add(os.path.basename(e.path))
    except Exception:
        pass  # no shards dir yet
    return out


def existing_hf_shards(api, repo_id):
    out = set()
    try:
        files = api.list_repo_files(repo_id, repo_type=REPO_TYPE)
    except Exception:
        return out
    for n in files:
        base = os.path.basename(n)
        if base.startswith("emb_") and base.endswith(".parquet"):
            out.add(base)
    return out


def resume_action(shard, vol_shards, hf_shards):
    on_vol = shard in vol_shards
    on_hf = shard in hf_shards
    if on_vol and on_hf:
        return "skip"
    if on_vol and not on_hf:
        return "push_from_vol"
    return "pack"


def _vol_read_retry(vol, name, dst, retries, delay=0.5):
    last = RuntimeError(f"no attempts made for {name}")
    for attempt in range(retries):
        try:
            with open(dst, "wb") as f:
                vol.read_file_into_fileobj(name, f)
            return dst
        except Exception as e:
            last = e
            time.sleep(delay * (2 ** attempt))
    raise last


def download_shard(vol, shard_names: list[str], stage_dir: str, workers: int = 6, retries: int = 3) -> list[str]:
    os.makedirs(stage_dir, exist_ok=True)

    def _one(name):
        dst = os.path.join(stage_dir, os.path.basename(name))
        return _vol_read_retry(vol, name, dst, retries)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, shard_names))


def upload_to_volume(vol, local_path: str, shard: str) -> None:
    with vol.batch_upload(force=True) as batch:
        batch.put_file(local_path, f"shards/{shard}")
    vol.commit()


def pull_volume_parquet(vol, shard: str, dst: str, retries: int = 3) -> None:
    _vol_read_retry(vol, f"shards/{shard}", dst, retries)


def push_to_hf(api, local_path: str, repo_id: str, shard: str) -> None:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"data/{shard}",
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message=f"Add {shard}",
    )


def _shard_name(i: int) -> str:
    return f"emb_{i:04d}.parquet"


def run_shard(vol, api, i, all_names, shard_rows, stage_dir, em_repo,
              workers=6, batch_size=64, retries=3):
    shard = _shard_name(i)
    chunk = all_names[i * shard_rows : (i + 1) * shard_rows]
    assert chunk, f"shard {i} has no rows"

    action = resume_action(shard, existing_volume_shards(vol), existing_hf_shards(api, em_repo))
    local_parquet = os.path.join(stage_dir, shard)
    os.makedirs(stage_dir, exist_ok=True)

    if action == "skip":
        print(f"[local-pack] shard {i}: {shard} already on volume+HF — skipping", flush=True)
        return action

    if action == "push_from_vol":
        pull_volume_parquet(vol, shard, local_parquet, retries)
    else:
        staged = download_shard(vol, chunk, stage_dir, workers=workers, retries=retries)
        pack_rows(iter_rows(staged), local_parquet, batch_size=batch_size)
        upload_to_volume(vol, local_parquet, shard)
        for p in staged:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    push_to_hf(api, local_parquet, em_repo, shard)
    try:
        os.remove(local_parquet)
    except FileNotFoundError:
        pass
    return action


def run_pipeline(vol, api, names, shard_rows, stage_dir, em_repo,
                 workers=6, batch_size=64, retries=3, lo=0, hi=None,
                 log=None, hf_only=False):
    """Pipelined variant of the run_shard loop.

    Overlaps network directions across shards: while shard i's HF push is
    uploading (fiber up), shard i+1's .pt download from the volume runs in a
    background thread (fiber down). Resume state is fetched once up front and
    updated incrementally instead of re-querying per shard. A failed shard
    aborts the run; rerunning resumes where it left off.

    hf_only=True skips the /data/shards volume copy — packed shards go to HF
    only. Halves upload traffic; the trainer reads .pt directly, so the volume
    copy is optional insurance (rehydratable from HF if ever needed)."""
    log = log or (lambda m: print(m, flush=True))
    n_shards = (len(names) + shard_rows - 1) // shard_rows
    hi = n_shards if hi is None else min(hi, n_shards)

    vol_shards = existing_volume_shards(vol)
    hf_shards = existing_hf_shards(api, em_repo)

    def chunk(i):
        return names[i * shard_rows:(i + 1) * shard_rows]

    def stage_of(i):
        # distinct from the parquet filename: this is a DIRECTORY holding the
        # shard's downloaded .pt files (avoids emb_XXXX.parquet collision)
        return os.path.join(stage_dir, _shard_name(i) + ".staged")

    dl = ThreadPoolExecutor(max_workers=1)

    def try_prefetch(i):
        """Start background download for shard i iff it will need packing."""
        if i >= hi or resume_action(_shard_name(i), vol_shards, hf_shards) != "pack":
            return None
        return dl.submit(download_shard, vol, chunk(i), stage_of(i), workers, retries)

    actions = []
    t0 = time.time()
    rows_done = 0
    os.makedirs(stage_dir, exist_ok=True)
    pending = try_prefetch(lo)
    for i in range(lo, hi):
        shard = _shard_name(i)
        action = resume_action(shard, vol_shards, hf_shards)
        n = len(chunk(i))
        assert n, f"shard {i} has no rows"
        local_parquet = os.path.join(stage_dir, shard)

        if action == "skip":
            log(f"[local-pack] shard {i}: {shard} already on volume+HF — skipping")
        else:
            if action == "push_from_vol":
                pull_volume_parquet(vol, shard, local_parquet, retries)
            else:
                staged = pending.result() if pending is not None else \
                    download_shard(vol, chunk(i), stage_of(i), workers, retries)
                pack_rows(iter_rows(staged), local_parquet, batch_size=batch_size)
                if not hf_only:
                    upload_to_volume(vol, local_parquet, shard)
                    vol_shards.add(shard)
                for p in staged:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            pending = try_prefetch(i + 1)  # overlap next down with this up
            push_to_hf(api, local_parquet, em_repo, shard)
            hf_shards.add(shard)
            try:
                os.remove(local_parquet)
            except FileNotFoundError:
                pass

        actions.append(action)
        rows_done += n
        elapsed = time.time() - t0
        rate = rows_done / max(1e-9, elapsed)
        eta = (len(names) - rows_done) / max(1e-9, rate) / 60
        log(f"[local-pack] done {rows_done}/{len(names)} ({100*rows_done/len(names):.0f}%)  "
            f"{rate:.0f} rows/s  ETA {eta:.0f} min  shard {i}/{hi} ({n} rows) action={action}")
    return actions


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-rows", type=int, default=SHARD_ROWS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stage-dir", default="/tmp/emb_stage")
    ap.add_argument("--em-repo", default=EMB_REPO)
    ap.add_argument("--only", default="", help="i[:j] shard range, e.g. 0 or 2:5")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--hf-only", action="store_true",
                    help="push packed shards to HF only; skip the /data/shards volume copy")
    args = ap.parse_args(argv)

    import modal
    from huggingface_hub import HfApi
    vol = modal.Volume.from_name(VOL_NAME)
    api = HfApi()

    entries = vol.listdir("embeddings")
    names = sorted_embedding_names(entries)
    if not names:
        print("[local-pack] no embeddings found under embeddings/ on the volume — aborting", flush=True)
        return
    slices = shard_slices(names, args.shard_rows)
    print(f"[local-pack] embeddings: {len(names)}  shards: {len(slices)}  "
          f"shard_rows={args.shard_rows}  workers={args.workers}", flush=True)

    lo, hi = 0, len(slices)
    if args.only:
        parts = args.only.split(":")
        lo = int(parts[0]) if parts[0] else 0
        hi = int(parts[1]) if len(parts) > 1 and parts[1] else len(slices)

    run_pipeline(vol, api, names, args.shard_rows,
                 args.stage_dir, args.em_repo,
                 workers=args.workers, batch_size=args.batch_size,
                 retries=args.retries, lo=lo, hi=hi, hf_only=args.hf_only)


if __name__ == "__main__":
    main()
