"""The pack must find the val rows by their GLOBAL row index.

The key index stores the row of an embedding as its position in the shard
(row_cursor + j, accumulated over row groups). The pack reads one row group
at a time, where `j` is the position WITHIN that group. Matching on `j`
alone finds only the rows that happen to sit in the first row group -- and
silently packs a biased subset of the val split.

Measured symptom on the live corpus: most shards reported "0/N rows" and a
few "N/N" -- exactly what a wrong row offset produces.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa
import pyarrow.parquet as pq

from vision_adapter.data.pack import SCHEMA


def test_a_row_in_the_second_row_group_is_found(tmp_path):
    shard = tmp_path / "emb_0009.parquet"
    # two row groups of 4 rows each (one write_table per group)
    w = pq.ParquetWriter(str(shard), SCHEMA, compression=None)
    try:
        w.write_table(pa.Table.from_pylist(
            [{"key": f"k{i}", "n_vis": 364, "vis_bytes": b"x" * 8} for i in range(4)],
            schema=SCHEMA))
        w.write_table(pa.Table.from_pylist(
            [{"key": f"k{i}", "n_vis": 364, "vis_bytes": b"x" * 8} for i in range(4, 8)],
            schema=SCHEMA))
    finally:
        w.close()
    assert pq.ParquetFile(shard).num_row_groups == 2

    import scripts.pack_val as pv

    class _Args:
        out_dir = str(tmp_path)
        cache_dir = str(tmp_path / "cache")

    # the wanted row is GLOBAL row 6 -> it lives in the SECOND row group
    chosen = [{"emb": "e6", "g": "agentic"}]
    index = {"e6": ("emb_0009.parquet", 6, 364)}
    import vision_adapter.data.stream as st

    orig = st._download_shard_hf_transfer
    st._download_shard_hf_transfer = lambda sf, cache_dir=None: str(shard)
    try:
        pv._write_pack(_Args(), "agentic", chosen, index, str(tmp_path / "cache"))
    finally:
        st._download_shard_hf_transfer = orig

    out = tmp_path / "val_pack_agentic.parquet"
    assert out.is_file(), "the pack must be written"
    tbl = pq.read_table(out)
    assert tbl.num_rows == 1, f"expected the row at global index 6, got {tbl.num_rows} rows"
    assert tbl.column("key").to_pylist() == ["k6"]
