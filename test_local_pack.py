# test_local_pack.py
import io
import numpy as np
import pyarrow.parquet as pq
import torch
from local_pack import (
    make_row,
    iter_rows,
    pack_rows,
    sorted_embedding_names,
    shard_slices,
    FileEntryLike,
    resume_action,
    existing_volume_shards,
    existing_hf_shards,
)


def _bf16(rows, cols=4096):
    return torch.randn(rows, cols, dtype=torch.bfloat16)


def test_make_row_shape_and_key():
    t = _bf16(5)
    r = make_row("embeddings/abc.pt", t)
    assert r["key"] == "embeddings/abc.pt"
    assert r["n_vis"] == 5
    assert len(r["vis_bytes"]) == 5 * 4096 * 2


def test_pack_rows_schema_roundtrip(tmp_path):
    t0, t1, t2 = _bf16(3), _bf16(7), _bf16(2)
    rows = [
        make_row("embeddings/a.pt", t0),
        make_row("embeddings/b.pt", t1),
        make_row("embeddings/c.pt", t2),
    ]
    out = tmp_path / "emb_0000.parquet"
    pack_rows(rows, str(out), batch_size=2)
    tbl = pq.read_table(str(out))
    assert tbl.num_rows == 3
    assert tbl.column_names == ["key", "n_vis", "vis_bytes"]
    row_b = tbl.slice(1, 1).to_pylist()[0]
    assert row_b["key"] == "embeddings/b.pt"
    assert row_b["n_vis"] == 7
    rt = (
        torch.from_numpy(np.frombuffer(row_b["vis_bytes"], dtype=np.uint8))
        .view(torch.bfloat16)
        .reshape(-1, 4096)
    )
    assert torch.equal(rt, t1)


def test_iter_rows_roundtrip(tmp_path):
    t = _bf16(4)
    p = tmp_path / "xyz.pt"
    torch.save(t, str(p))
    r = make_row("embeddings/xyz.pt", t)
    packed = list(iter_rows([str(p)]))
    assert packed[0]["key"] == "embeddings/xyz.pt"
    assert packed[0]["n_vis"] == 4
    rt = (
        torch.from_numpy(np.frombuffer(packed[0]["vis_bytes"], dtype=np.uint8))
        .view(torch.bfloat16)
        .reshape(-1, 4096)
    )
    assert torch.equal(rt, t)


def test_sorted_embedding_names_is_sorted_and_strips_prefix():
    entries = [
        FileEntryLike(path="embeddings/zzz.pt"),
        FileEntryLike(path="embeddings/aaa.pt"),
        FileEntryLike(path="embeddings/mmm.pt"),
    ]
    assert sorted_embedding_names(entries) == [
        "embeddings/aaa.pt",
        "embeddings/mmm.pt",
        "embeddings/zzz.pt",
    ]


def test_shard_slices_boundaries():
    names = [f"embeddings/{i:04d}.pt" for i in range(7)]
    s = shard_slices(names, 3)
    assert s == [
        ["embeddings/0000.pt", "embeddings/0001.pt", "embeddings/0002.pt"],
["embeddings/0003.pt", "embeddings/0004.pt", "embeddings/0005.pt"],
        ["embeddings/0006.pt"],
    ]


def test_resume_action_skip_pack_push_from_vol():
    assert resume_action("emb_0002.parquet", {"emb_0002.parquet"}, {"emb_0002.parquet"}) == "skip"
    assert resume_action("emb_0002.parquet", set(), set()) == "pack"
    assert resume_action("emb_0002.parquet", {"emb_0002.parquet"}, set()) == "push_from_vol"
    assert resume_action("emb_0002.parquet", set(), {"emb_0002.parquet"}) == "pack"


def _fakedisk():
    class FakeVol:
        def __init__(self, vol_paths):
            self._paths = vol_paths

        def listdir(self, path):
            prefix = path.rstrip("/") + "/"
            return [FileEntryLike(p) for p in self._paths if p.startswith(prefix)]

    return FakeVol


def _fakeapi(files):
    class FakeApi:
        def list_repo_files(self, repo_id, repo_type=None):
            return files

    return FakeApi


def test_existing_volume_shards_strips_prefix(tmp_path):
    vol = _fakedisk()(["shards/emb_0000.parquet", "shards/emb_0001.parquet", "shards/.keep"])
    assert existing_volume_shards(vol) == {"emb_0000.parquet", "emb_0001.parquet"}


def test_existing_hf_shards_strips_prefix():
    api = _fakeapi(["data/emb_0000.parquet", "data/emb_0001.parquet", "dataset_infos.json", ".gitattributes"])()
    assert existing_hf_shards(api, "keypa/vision-adapter-embeddings") == {
        "emb_0000.parquet",
        "emb_0001.parquet",
    }
