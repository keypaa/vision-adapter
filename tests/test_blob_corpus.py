"""The blob corpus format: exact bytes, a tiny index, and coalesced ranges.

The format is what makes "fetch 1% of the corpus" cost 1% of the bytes
(docs/DATA_PATH_REBUILD.md D2): a row is the unit of transfer, its offset is
a prefix sum, and the index is n_vis per row.
"""
import json

import numpy as np
import pytest

from vision_adapter.data.blob_corpus import (
    ROW_BYTES,
    BlobCorpus,
    ShardInfo,
    write_corpus_meta,
)


def _write_corpus(tmp_path, n_shards=2, rows_per_shard=5, n_vis=(3, 7, 11, 1, 5)):
    """A tiny corpus: rows of known n_vis, written in the blob format."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shards = []
    keys, shard_col, row_col, nvis_col, grid_col = [], [], [], [], []
    for si in range(n_shards):
        name = f"emb_{si:04d}"
        blob = bytearray()
        idx = []
        for r in range(rows_per_shard):
            nv = n_vis[r % len(n_vis)]
            payload = bytes([(si * 31 + r) % 256]) * (nv * ROW_BYTES)
            assert len(payload) == nv * ROW_BYTES
            blob += payload
            idx.append(nv)
            keys.append(f"embeddings/k{si}_{r}.pt")
            shard_col.append(si)
            row_col.append(r)
            nvis_col.append(nv)
            grid_col.append([1, nv * 2, 2])
        (tmp_path / f"{name}.bin").write_bytes(bytes(blob))
        (tmp_path / f"{name}.idx").write_bytes(np.array(idx, dtype=np.uint16).tobytes())
        shards.append(ShardInfo(f"{name}.bin", rows_per_shard, len(blob)))
    pq.write_table(pa.table({"key": keys, "shard": shard_col, "row": row_col,
                              "n_vis": nvis_col, "grid_thw": grid_col}),
                   tmp_path / "rows.parquet")
    write_corpus_meta(tmp_path / "corpus.json", shards, cap_patches=1531,
                      source="test")
    return shards


def test_a_row_reads_back_exactly(tmp_path):
    _write_corpus(tmp_path)
    c = BlobCorpus.open(str(tmp_path))
    for si in range(2):
        for r in range(5):
            nv = c.n_vis(si, r)
            raw = c.row_bytes(si, r)
            assert len(raw) == nv * ROW_BYTES
            expect = bytes([(si * 31 + r) % 256]) * (nv * ROW_BYTES)
            assert raw == expect, (si, r)


def test_the_index_is_a_prefix_sum_of_the_rows(tmp_path):
    shards = _write_corpus(tmp_path)
    c = BlobCorpus.open(str(tmp_path))
    off, n = c.row_range(0, 0)
    assert off == 0
    for r in range(5):
        lo, ln = c.row_range(0, r)
        assert lo == off
        assert ln == c.n_vis(0, r) * ROW_BYTES
        off += ln
    assert off == shards[0].bytes


def test_a_set_of_rows_coalesces_into_one_range(tmp_path):
    _write_corpus(tmp_path)
    c = BlobCorpus.open(str(tmp_path))
    lo, ln = c.rows_range(0, [1, 2, 3])
    assert ln == sum(c.n_vis(0, r) * ROW_BYTES for r in (1, 2, 3))
    # the range covers exactly those rows
    blob = c.shard_bytes(0)[lo:lo + ln]
    assert len(blob) == ln


def test_the_key_map_and_the_geometry_survive(tmp_path):
    _write_corpus(tmp_path)
    c = BlobCorpus.open(str(tmp_path))
    assert len(c) == 10
    sh, rw, nv = c.lookup("embeddings/k1_3.pt")
    assert (sh, rw) == (1, 3)
    assert nv == c.n_vis(sh, rw)
    assert c.grid_thw(sh, rw) == [1, c.n_vis(sh, rw) * 2, 2]
    assert "embeddings/k0_0.pt" in c
    with pytest.raises(KeyError):
        c.lookup("embeddings/nope.pt")


def test_the_row_tensor_has_the_right_shape(tmp_path):
    _write_corpus(tmp_path)
    c = BlobCorpus.open(str(tmp_path))
    t = c.row_tensor(0, 1)
    assert t.shape == (c.n_vis(0, 1), 4096)


def test_the_index_is_tiny(tmp_path):
    """The whole point: the index is n_vis per row, not a parquet footer."""
    shards = _write_corpus(tmp_path, n_shards=2, rows_per_shard=5)
    idx_bytes = (tmp_path / "emb_0000.idx").stat().st_size
    assert idx_bytes == 5 * 2          # uint16 per row
    assert idx_bytes * 100 < shards[0].bytes


def test_corpus_json_describes_the_corpus(tmp_path):
    _write_corpus(tmp_path)
    meta = json.loads((tmp_path / "corpus.json").read_text())
    assert meta["version"] == 2
    assert meta["format"] == "blob+idx"
    assert meta["total_rows"] == 10
    assert meta["cap_patches"] == 1531
    assert len(meta["shards"]) == 2


def test_read_range_returns_exactly_the_asked_bytes(tmp_path):
    _write_corpus(tmp_path)
    c = BlobCorpus.open(str(tmp_path))
    off, ln = c.row_range(0, 2)
    raw = c.read_range(0, off, ln)
    assert raw == c.row_bytes(0, 2)
    assert len(raw) == ln


def test_the_row_plan_is_row_granular_and_deterministic(tmp_path):
    from vision_adapter.data.blob_corpus import build_row_plan

    _write_corpus(tmp_path, n_shards=2, rows_per_shard=5)
    c = BlobCorpus.open(str(tmp_path))
    keys = [f"embeddings/k{s}_{r}.pt" for s in range(2) for r in range(5)]
    plan = build_row_plan(c, keys, sample_size=0, seed=0)
    assert len(plan) == 10
    assert plan == build_row_plan(c, keys, sample_size=0, seed=0), "deterministic"
    small = build_row_plan(c, keys, sample_size=3, seed=1)
    assert len(small) == 3
    assert small == build_row_plan(c, keys, sample_size=3, seed=1)
    total = sum(c.n_vis(s, r) * ROW_BYTES for s, r in small)
    assert total < sum(c.n_vis(s, r) * ROW_BYTES for s, r in plan)


def test_the_stream_dataset_serves_the_plan(tmp_path):
    from vision_adapter.data.blob_corpus import BlobStreamDataset, build_row_plan

    _write_corpus(tmp_path, n_shards=2, rows_per_shard=5)
    c = BlobCorpus.open(str(tmp_path))
    keys = [f"embeddings/k{s}_{r}.pt" for s in range(2) for r in range(5)]
    plan = build_row_plan(c, keys, sample_size=0, seed=0)
    rows = list(BlobStreamDataset(c, plan, group=4))
    assert len(rows) == len(plan)
    s, r = plan[0]
    assert rows[0].shape == (c.n_vis(s, r), 4096)
    assert "bfloat16" in str(rows[0].dtype)


def test_the_stream_dataset_skips_the_resume_offset(tmp_path):
    from vision_adapter.data.blob_corpus import BlobStreamDataset, build_row_plan

    _write_corpus(tmp_path, n_shards=2, rows_per_shard=5)
    c = BlobCorpus.open(str(tmp_path))
    keys = [f"embeddings/k{s}_{r}.pt" for s in range(2) for r in range(5)]
    plan = build_row_plan(c, keys, sample_size=0, seed=0)
    rows = list(BlobStreamDataset(c, plan, start_pos=3, group=4))
    assert len(rows) == len(plan) - 3
    s, r = plan[3]
    assert rows[0].shape == (c.n_vis(s, r), 4096)
