from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

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


def iter_rows(local_paths):
    """torch.load each staged .pt, yield a row dict (key = embeddings/<basename>)."""
    for p in local_paths:
        t = torch.load(p, map_location="cpu", weights_only=True)
        yield make_row(f"embeddings/{os.path.basename(p)}", t)


def pack_rows(rows, out_path, batch_size=64):
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