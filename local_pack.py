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
    """Return sorted `embeddings/<sha1>.pt` paths (matches Modal's sorted(glob))."""
    return sorted(e.path for e in entries if e.path.startswith("embeddings/"))


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
    """Stream rows to a parquet file in fixed-size batches (RAM-bounded)."""
    writer = pq.ParquetWriter(out_path, SCHEMA)
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
