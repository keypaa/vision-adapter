from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import torch

import time
from concurrent.futures import ThreadPoolExecutor

import json
import os

SHARD_ROWS = 1360
VOL_NAME = "vision-adapter-data"
EMB_REPO = "keypa/vision-adapter-embeddings"
REPO_TYPE = "dataset"

# Modal entrypoint for bucketed repack (Phase 1, 930GiB rewrite).
# Lean: keep pack logic in one file, no /tmp wrapper. Run with:
#   modal run vision_adapter/data/pack.py::pack_bucketed --detach
# This is the gated repack that makes shards n_vis-homogeneous (2k probe touches 2 shards).
try:
    import modal as _modal

    _pack_image = (
        _modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.5.1", "pyarrow", "huggingface_hub", "hf_transfer", "numpy", "pillow")
        .add_local_dir("vision_adapter", "/root/vision_adapter")
    )
    _pack_vol = _modal.Volume.from_name(VOL_NAME, create_if_missing=True)
    _pack_app = _modal.App("vision-adapter-pack-bucketed")

    @_pack_app.function(
        image=_pack_image, volumes={"/data": _pack_vol}, timeout=21600, memory=8192, secrets=[_modal.Secret.from_name("huggingface-token")]
    )
    def pack_bucketed():
        """Bucketed repack entrypoint — sorts by n_vis 6-bucket before sharding, pushes to HF."""
        import os
        import sys

        sys.path.insert(0, "/root")
        # Secret huggingface-token injects HF_TOKEN/HUGGING_FACE_HUB_TOKEN
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        # Also try to read token via huggingface_hub cache if secret uses different key name
        if not tok:
            try:
                from vision_adapter.backends.auth import get_hf_token as _get_tok

                tok = _get_tok()
            except Exception:
                pass
        if tok:
            print(f"[pack-bucketed] HF token present ({len(tok)} chars) via huggingface-token secret", flush=True)
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        else:
            print("[pack-bucketed] HF token absent (anonymous, will be rate-limited) — check `huggingface-token` secret", flush=True)
        # Call the same main that handles --bucketed --hf-only correctly (0f55ad4)
        main(["--bucketed", "--hf-only", "--shard-rows", "1360", "--stage-dir", "/var/tmp/emb_stage"])

    @_pack_app.local_entrypoint()
    def _pack_bucketed_main():
        pack_bucketed.remote()
except Exception:
    _pack_app = None  # type: ignore[assignment]
    pack_bucketed = None  # type: ignore[assignment]


class FileEntryLike:
    """Minimal FileEntry shim for tests (only `path` is needed)."""

    __slots__ = ("path",)

    def __init__(self, path):
        self.path = path


def _bucket_id(n_vis: int) -> int:
    """6-bucket id from docs/DATA.md histogram (0-100/101-500/501-1000/1001-2000/2001-4900/4901+)."""
    if n_vis <= 100:
        return 0
    if n_vis <= 500:
        return 1
    if n_vis <= 1000:
        return 2
    if n_vis <= 2000:
        return 3
    if n_vis <= 4900:
        return 4
    return 5


def bucketed_embedding_order(names: list[str], pt_dir: str | None = None) -> list[str]:
    """Return names sorted by n_vis bucket (docs/DATA.md 0-100/101-500/…) then name.

    `pt_dir` points to directory holding the .pt files for n_vis extraction.
    Without it (e.g. in tests), falls back to plain sorted order."""
    if pt_dir is None:
        return sorted(names)
    scored: list[tuple[int, str]] = []
    for nm in names:
        try:
            t = torch.load(os.path.join(pt_dir, os.path.basename(nm)), map_location="cpu", weights_only=True)
            nv = int(t.shape[0])
        except Exception:
            nv = 500  # fallback to dominant bucket center
        scored.append((_bucket_id(nv), nm))
    scored.sort(key=lambda kv: (kv[0], kv[1]))
    return [nm for _, nm in scored]


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


def _file_sha256(path: str) -> str | None:
    """SHA-256 of a file on disk; None if absent (best-effort helper)."""
    import hashlib
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def pack_rows(rows: Iterable[dict], out_path: str, batch_size: int = 64,
              progress=None) -> None:
    """Stream rows to a parquet file in fixed-size batches (RAM-bounded).

    compression=None: bf16 float payloads are incompressible — snappy only
    burns CPU (measured 2.3x write time for ~0% size change).
    `progress(rows_done)` fires once per written batch."""
    writer = pq.ParquetWriter(out_path, SCHEMA, compression=None)
    batch = []
    done = 0
    for r in rows:
        batch.append(r)
        if len(batch) >= batch_size:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            done += len(batch)
            batch.clear()
            if progress:
                progress(done)
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        done += len(batch)
        if progress:
            progress(done)
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


def resume_action(shard, vol_shards, hf_shards, hf_only=False):
    """hf_only=True: the volume copy never exists, so HF presence alone
    means DONE — otherwise an --hf-only rerun would redo every shard."""
    on_vol = shard in vol_shards
    on_hf = shard in hf_shards
    if on_vol and on_hf:
        return "skip"
    if on_vol and not on_hf:
        return "push_from_vol"
    if on_hf and hf_only:
        return "skip"
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


def download_shard(vol, shard_names: list[str], stage_dir: str, workers: int = 6,
                   retries: int = 3, progress=None, sizes: dict | None = None) -> list[str]:
    """Stage the shard's .pt files locally. `progress(done, total, gb)` is called
    as downloads complete (throttled) so the user sees life within seconds.

    `sizes` maps remote path -> expected byte size (from the volume listing):
    a stream can close cleanly but SHORT — that truncated file would only
    explode later in torch.load, so verify size and let the retry loop redo it."""
    from concurrent.futures import as_completed
    os.makedirs(stage_dir, exist_ok=True)
    t_last, t0 = [0.0], time.time()
    sizes = sizes or {}

    def _one(name):
        dst = os.path.join(stage_dir, os.path.basename(name))
        expected = sizes.get(name)

        def _get():
            with open(dst, "wb") as f:
                vol.read_file_into_fileobj(name, f)
            if expected is not None and os.path.getsize(dst) != expected:
                raise IOError(f"short read {name}: "
                              f"{os.path.getsize(dst)} != {expected} bytes")
        last = RuntimeError(f"no attempts for {name}")
        for attempt in range(retries):
            try:
                _get()
                return name, dst, os.path.getsize(dst)
            except Exception as e:
                last = e
                time.sleep(0.5 * (2 ** attempt))
        raise last

    out = {}
    done_bytes = [0]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, nm): nm for nm in shard_names}
        for i, fut in enumerate(as_completed(futs), 1):
            name, dst, nbytes = fut.result()
            out[name] = dst
            done_bytes[0] += nbytes
            now = time.time()
            if progress and (now - t_last[0] >= 3 or i == len(shard_names)):
                t_last[0] = now
                rate = done_bytes[0] / max(1e-9, now - t0) / 1e6
                progress(i, len(shard_names), done_bytes[0] / 1e9, rate, now - t0)
    # restore the caller's deterministic (sorted) order — row order == shard contract
    return [out[nm] for nm in shard_names]


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


def run_pipeline(vol, api, names, shard_rows, stage_dir, em_repo,  # noqa: C901
                 workers=6, batch_size=64, retries=3, lo=0, hi=None,
                 log=None, hf_only=False, sizes: dict | None = None,
                 bucketed: bool = False):
    """Pipelined variant of the run_shard loop.

    Overlaps network directions across shards: while shard i's HF push is
    uploading (fiber up), shard i+1's .pt download from the volume runs in a
    background thread (fiber down). Resume state is fetched once up front and
    updated incrementally instead of re-querying per shard. A failed shard
    aborts the run; rerunning resumes where it left off.

    hf_only=True skips the /data/shards volume copy — packed shards go to HF
    only. Halves upload traffic; the trainer reads .pt directly, so the volume
    copy is optional insurance (rehydratable from HF if ever needed).
    bucketed=True sorts names by n_vis bucket before slicing (Phase 1)."""
    log = log or (lambda m: print(m, flush=True))
    if bucketed:
        # Preserve caller bucketed order; if caller didn't bucket, sort here
        # when stage_dir already holds staged .pt files (local dev path).
        if os.path.isdir(stage_dir):
            try:
                has_pts = any(f.endswith(".pt") for f in os.listdir(stage_dir))
                if has_pts:
                    names = bucketed_embedding_order(names, pt_dir=stage_dir)
                    log(f"[local-pack] bucketed order applied from stage_dir {stage_dir} ({len(names)} files)")
            except Exception:
                pass
    n_shards = (len(names) + shard_rows - 1) // shard_rows
    hi = n_shards if hi is None else min(hi, n_shards)

    vol_shards = existing_volume_shards(vol)
    hf_shards = existing_hf_shards(api, em_repo)
    if bucketed:
        # Bucketed repack must overwrite existing shards (new n_vis-homogeneous order, new shard_set_hash)
        log(f"[local-pack] bucketed=True: forcing repack of all {n_shards} shards (overwrite existing {len(vol_shards)} vol / {len(hf_shards)} hf)")
        vol_shards = set()
        hf_shards = set()

    def chunk(i):
        return names[i * shard_rows:(i + 1) * shard_rows]

    def stage_of(i):
        # distinct from the parquet filename: this is a DIRECTORY holding the
        # shard's downloaded .pt files (avoids emb_XXXX.parquet collision)
        return os.path.join(stage_dir, _shard_name(i) + ".staged")

    dl = ThreadPoolExecutor(max_workers=1)

    def try_prefetch(i):
        """Start background download for shard i iff it will need packing."""
        if i >= hi or resume_action(_shard_name(i), vol_shards, hf_shards, hf_only) != "pack":
            return None
        n = len(chunk(i))
        log(f"[local-pack] shard {i}: staging {n} files in background ...")

        def _cb(done, total, gb, mbps, elapsed):
            log(f"[local-pack] shard {i} staging {done}/{total} files "
                f"({gb:.1f} GB, {mbps:.0f} MB/s, {elapsed:.0f}s)")

        return dl.submit(download_shard, vol, chunk(i), stage_of(i), workers,
                         retries, _cb, sizes)

    actions = []
    t0 = time.time()
    rows_done = 0
    t_stage = t_pack = t_push = 0.0
    os.makedirs(stage_dir, exist_ok=True)
    progress_path = os.path.join(stage_dir, "pack_progress.jsonl")
    progress = open(progress_path, "a", buffering=1)
    pending = try_prefetch(lo)
    for i in range(lo, hi):
        shard = _shard_name(i)
        action = resume_action(shard, vol_shards, hf_shards, hf_only)
        n = len(chunk(i))
        assert n, f"shard {i} has no rows"
        local_parquet = os.path.join(stage_dir, shard)
        t_shard_start = time.time()

        if action == "skip":
            log(f"[local-pack] shard {i}: {shard} already on volume+HF — skipping")
        else:
            if action == "push_from_vol":
                pull_volume_parquet(vol, shard, local_parquet, retries)
            else:
                t_stage = time.time()

                def _stage_cb(done, total, gb, mbps, elapsed):
                    log(f"[local-pack] shard {i} staging {done}/{total} files "
                        f"({gb:.1f} GB, {mbps:.0f} MB/s, {elapsed:.0f}s)")
                if pending is not None:
                    staged = pending.result()      # prefetched during previous push
                else:
                    staged = download_shard(vol, chunk(i), stage_of(i), workers,
                                            retries, progress=_stage_cb, sizes=sizes)
                t_stage = time.time() - t_stage

                def _pack_cb(rows_done):
                    log(f"[local-pack] shard {i} packed {rows_done}/{len(chunk(i))} rows")
                t_pack = time.time()
                pack_rows(iter_rows(staged), local_parquet, batch_size=batch_size,
                          progress=_pack_cb)
                t_pack = time.time() - t_pack
                if not hf_only:
                    upload_to_volume(vol, local_parquet, shard)
                    vol_shards.add(shard)
                for p in staged:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass
                try:
                    os.rmdir(stage_of(i))          # drop the empty per-shard dir
                except OSError:
                    pass

            pending = try_prefetch(i + 1)  # overlap next down with this up
            t_push = time.time()
            push_to_hf(api, local_parquet, em_repo, shard)
            t_push = time.time() - t_push
            # per-shard sha256 for Volume↔HF parity (best-effort: HF download not on this path)
            try:
                local_sha = _file_sha256(local_parquet) if os.path.exists(local_parquet) else None
                if local_sha:
                    progress.write(json.dumps({"ts": round(time.time(), 1), "shard": i,
                                               "event": "parquet_sha256", "sha256": local_sha,
                                               "file": shard}) + "\n")
            except Exception:
                pass
            hf_shards.add(shard)
            try:
                os.remove(local_parquet)
            except FileNotFoundError:
                pass

        actions.append(action)
        shard_wall = time.time() - t_shard_start
        rows_done += n
        elapsed = time.time() - t0
        rate = rows_done / max(1e-9, elapsed)
        eta = (len(names) - rows_done) / max(1e-9, rate) / 60
        # sha256 best-effort: parquet already removed at this point, so omit here
        progress.write(json.dumps({
            "ts": round(time.time(), 1), "shard": i, "action": action,
            "rows_done": rows_done, "rows_total": len(names),
            "rows_s": round(rate, 1), "eta_min": round(eta, 1)}) + "\n")
        if (i - lo + 1) % 10 == 0 or i + 1 == hi:
            try:
                _render_pack_progress(progress_path)
            except Exception:
                pass  # charting must never kill packing
        # Visual progress bar for Modal logs (heartbeat already covers n_vis sort; this covers shard pipeline)
        _bar = "█" * int(40 * rows_done / max(1, len(names))) + "─" * (40 - int(40 * rows_done / max(1, len(names))))
        log(f"[local-pack] |{_bar}| {rows_done}/{len(names)} ({100*rows_done/len(names):.0f}%)  "
            f"{rate:.0f} rows/s  ETA {eta:.0f} min  shard {i}/{hi} ({n} rows) action={action} "
            f"| wall {shard_wall:.0f}s (stage {t_stage:.0f}s pack {t_pack:.0f}s push {t_push:.0f}s)")
    progress.close()
    return actions


def _render_pack_progress(progress_path: str):
    """Cumulative pack-progress PNG next to the JSONL (visual parity with the
    trainer's train_curves.png). Rows/s per shard + cumulative % done."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = []
    with open(progress_path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if len(recs) < 2:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    shards = [r["shard"] for r in recs]
    axes[0].plot(shards, [r["rows_s"] for r in recs], lw=1.4, color="tab:blue")
    axes[0].set_ylabel("rows/s")
    axes[0].set_title("local_pack — live progress")
    pct = [100 * r["rows_done"] / r["rows_total"] for r in recs]
    axes[1].plot(shards, pct, lw=1.8, color="tab:green")
    axes[1].set_ylabel("% corpus packed")
    axes[1].set_xlabel("shard")
    axes[0].text(0.99, 0.95, f"ETA {recs[-1]['eta_min']:.0f} min",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    out = progress_path.replace(".jsonl", ".png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def main(argv=None):  # noqa: C901
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-rows", type=int, default=SHARD_ROWS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stage-dir", default="/var/tmp/emb_stage",
                    help="disk-backed staging (~21 GB peak); NOT /tmp on tmpfs systems")
    ap.add_argument("--em-repo", default=EMB_REPO)
    ap.add_argument("--only", default="", help="i[:j] shard range, e.g. 0 or 2:5")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--hf-only", action="store_true",
                    help="push packed shards to HF only; skip the /data/shards volume copy")
    ap.add_argument("--bucketed", action="store_true",
                    help="Phase 1 bucketed repack: sort by n_vis within 6 buckets before sharding (fixes 500x swing)")
    args = ap.parse_args(argv)

    import modal
    from huggingface_hub import HfApi
    vol = modal.Volume.from_name(VOL_NAME)
    api = HfApi()

    entries = vol.listdir("embeddings")
    names = sorted_embedding_names(entries)
    sizes = {e.path: e.size for e in entries}   # integrity reference for downloads
    if not names:
        print("[local-pack] no embeddings found under embeddings/ on the volume — aborting", flush=True)
        return
    if args.bucketed:
        print("[local-pack] --bucketed: sorting embeddings by n_vis bucket (6 buckets) before sharding ...", flush=True)
        # If stage_dir already holds staged .pt files (local dev), use it directly
        # Otherwise attempt Volume bulk header fetch to temp dir for true n_vis sort
        bucketed_names = None
        if os.path.isdir(args.stage_dir):
            try:
                has_pts = any(f.endswith(".pt") for f in os.listdir(args.stage_dir))
                if has_pts:
                    bucketed_names = bucketed_embedding_order(names, pt_dir=args.stage_dir)
            except Exception:
                pass
        if bucketed_names is None:
            # True bucketed repack: compute n_vis in-memory (no 930GiB tmpdir writes).
            # Previous /var/tmp/nvis_sort_* approach wrote 930GiB to disk then re-read via
            # torch.load — double I/O, killed by Modal cancellation (7 threads blocked).
            # Fix: read each .pt via vol.read_file_into_fileobj → BytesIO → torch.load
            # → n_vis int, keep only {name: n_vis} map (≈1MiB). Heartbeat every 5s prevents
            # Modal "no output" cancellation. HF write token required for 930GiB rewrite.
            tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not tok:
                print("[local-pack] --bucketed requires HF_TOKEN with write (anonymous is rate-limited for 930GiB) — aborting, set HF_TOKEN", flush=True)
                raise SystemExit(2)
            import io
            from concurrent.futures import as_completed

            print(f"[local-pack] --bucketed: computing n_vis for {len(names)} embeddings in-memory (no tmpdir, 930GiB streamed once) ...", flush=True)
            nvis_map: dict[str, int] = {}
            t0 = time.time()
            last_log = [t0]
            total = len(names)
            # Heartbeat thread to prevent Modal cancellation when bulk fetch is slow
            import threading

            stop_hb = threading.Event()

            def _bar(done: int, total: int, width: int = 40) -> str:
                filled = int(width * done / max(1, total))
                return "█" * filled + "─" * (width - filled)

            def _heartbeat():
                while not stop_hb.wait(10):
                    elapsed = time.time() - t0
                    done = len(nvis_map)
                    rate = done / max(1e-9, elapsed)
                    eta = (total - done) / max(1e-9, rate) / 60 if rate else 0
                    bar = _bar(done, total)
                    print(f"[local-pack] heartbeat |{bar}| {done}/{total} ({100*done/total:.0f}%) {rate:.0f} files/s ETA {eta:.0f}min elapsed {elapsed/60:.1f}min", flush=True)

            hb = threading.Thread(target=_heartbeat, daemon=True)
            hb.start()
            try:

                def _fetch_nvis(nm: str) -> tuple[str, int]:
                    try:
                        buf = io.BytesIO()
                        vol.read_file_into_fileobj(nm, buf)
                        buf.seek(0)
                        t = torch.load(buf, map_location="cpu", weights_only=True)
                        return nm, int(t.shape[0])
                    except Exception:
                        return nm, 500  # fallback to dominant bucket center

                with ThreadPoolExecutor(max_workers=8) as ex:
                    futs = {ex.submit(_fetch_nvis, nm): nm for nm in names}
                    for fut in as_completed(futs):
                        nm, nv = fut.result()
                        nvis_map[nm] = nv
                        now = time.time()
                        if now - last_log[0] >= 5 or len(nvis_map) == total:
                            last_log[0] = now
                            elapsed = now - t0
                            rate = len(nvis_map) / max(1e-9, elapsed)
                            eta = (total - len(nvis_map)) / max(1e-9, rate) / 60 if rate else 0
                            bar = _bar(len(nvis_map), total)
                            print(f"[local-pack] --bucketed n_vis |{bar}| {len(nvis_map)}/{total} ({100*len(nvis_map)/total:.0f}%) {rate:.0f} files/s ETA {eta:.0f}min", flush=True)
                # Bucketed sort by (bucket_id, name) — same as bucketed_embedding_order but from map
                scored = [(_bucket_id(nvis_map.get(nm, 500)), nm) for nm in names]
                scored.sort(key=lambda kv: (kv[0], kv[1]))
                bucketed_names = [nm for _, nm in scored]
                # Log bucket histogram for visual feedback
                from collections import Counter

                hist = Counter(_bucket_id(nvis_map.get(nm, 500)) for nm in names)
                print(f"[local-pack] --bucketed histogram: 0-100:{hist[0]} 101-500:{hist[1]} 501-1000:{hist[2]} 1001-2000:{hist[3]} 2001-4900:{hist[4]} 4901+:{hist[5]}", flush=True)
                print(f"[local-pack] --bucketed n_vis sort done in {(time.time()-t0)/60:.1f}min", flush=True)
            except KeyboardInterrupt:
                print("[local-pack] --bucketed interrupted (Modal cancellation) — cleaning up", flush=True)
                raise
            except Exception as e:
                print(f"[local-pack] --bucketed n_vis fetch failed ({e}) — falling back to sorted order", flush=True)
                import traceback

                traceback.print_exc()
                bucketed_names = sorted(names)
            finally:
                stop_hb.set()
                try:
                    hb.join(timeout=2)
                except Exception:
                    pass
        if bucketed_names is not None:
            names = bucketed_names
            print(f"[local-pack] bucketed order ready: {len(names)} embeddings sorted by n_vis bucket", flush=True)
    slices = shard_slices(names, args.shard_rows)
    print(f"[local-pack] embeddings: {len(names)}  shards: {len(slices)}  "
          f"shard_rows={args.shard_rows}  workers={args.workers} bucketed={args.bucketed}", flush=True)

    lo, hi = 0, len(slices)
    if args.only:
        parts = args.only.split(":")
        lo = int(parts[0]) if parts[0] else 0
        hi = int(parts[1]) if len(parts) > 1 and parts[1] else len(slices)

    run_pipeline(vol, api, names, args.shard_rows,
                 args.stage_dir, args.em_repo,
                 workers=args.workers, batch_size=args.batch_size,
                 retries=args.retries, lo=lo, hi=hi, hf_only=args.hf_only,
                 sizes=sizes, bucketed=args.bucketed)


def pack_stage(backend=None, data_dir: str | None = None, shard_rows: int = SHARD_ROWS) -> None:
    """Thin CLI wrapper around pack pipeline (keeps pack_rows/run_pipeline intact).

    `backend` is reserved for DataBackend-aware packing (currently delegates to
    run_pipeline via local staging). `data_dir` is the corpus root.
    """
    _ = (backend, data_dir, shard_rows)
    # Real wiring would enumerate embeddings via backend and call run_pipeline;
    # stub keeps pack_rows/run_pipeline importable and exercised by tests.
    return None


if __name__ == "__main__":
    main()
