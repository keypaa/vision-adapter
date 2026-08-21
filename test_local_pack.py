# test_local_pack.py
import io
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch as _torch  # keep test-local alias
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
    download_shard,
    upload_to_volume,
    push_to_hf,
    pull_volume_parquet,
    run_shard,
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


def test_sorted_embedding_names_tolerates_bare_basenames():
    entries = [
        FileEntryLike(path="zzz.pt"),
        FileEntryLike(path="aaa.pt"),
        FileEntryLike(path="sub/other.pt"),
    ]
    assert sorted_embedding_names(entries) == [
        "embeddings/aaa.pt",
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


class FakeVol:
    """In-memory volume: read_file_into_fileobj + batch_upload + listdir."""

    def __init__(self, files=None):
        self.files = files or {}          # path -> bytes
        self.uploaded = {}               # shard -> local_path
        self.committed = 0

    def read_file_into_fileobj(self, path, fileobj, progress_cb=None):
        fileobj.write(self.files[path])
        return len(self.files[path])

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        return [FileEntryLike(p) for p in self.files if p.startswith(prefix)]

    class _Batch:
        def __init__(self, outer): self.outer = outer
        def put_file(self, local, remote): self.outer.uploaded[os.path.basename(remote)] = local
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def batch_upload(self, force=False):
        return FakeVol._Batch(self)

    def commit(self):
        self.committed += 1


class FakeApi:
    def __init__(self):
        self.calls = []
    def list_repo_files(self, repo_id, repo_type=None):
        return []
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
        self.calls.append((path_or_fileobj, path_in_repo, repo_id))
        return "ok"


def _make_pt(path, tensor):
    _torch.save(tensor, path)
    return open(path, "rb").read()


def test_download_shard_writes_files(tmp_path):
    vol = FakeVol()
    buf = io.BytesIO(); _torch.save(_bf16(3), buf); data = buf.getvalue()
    vol.files["embeddings/a.pt"] = data
    vol.files["embeddings/b.pt"] = data
    out = download_shard(vol, ["embeddings/a.pt", "embeddings/b.pt"], str(tmp_path / "stage"), workers=1)
    assert len(out) == 2
    assert (tmp_path / "stage" / "a.pt").read_bytes() == data


def test_upload_to_volume_records_shard(tmp_path):
    vol = FakeVol()
    p = tmp_path / "emb_0000.parquet"
    p.write_bytes(b"parquet-bytes")
    upload_to_volume(vol, str(p), "emb_0000.parquet")
    assert vol.uploaded["emb_0000.parquet"] == str(p)


def test_push_to_hf_invokes_api(tmp_path):
    api = FakeApi()
    p = tmp_path / "emb_0000.parquet"
    p.write_bytes(b"x")
    push_to_hf(api, str(p), "keypa/vision-adapter-embeddings", "emb_0000.parquet")
    assert api.calls == [(str(p), "data/emb_0000.parquet", "keypa/vision-adapter-embeddings")]


def test_pull_volume_parquet_writes_local(tmp_path):
    vol = FakeVol()
    vol.files["shards/emb_0099.parquet"] = b"parquet-bytes"
    dst = tmp_path / "pulled.parquet"
    pull_volume_parquet(vol, "emb_0099.parquet", str(dst))
    assert dst.read_bytes() == b"parquet-bytes"


def test_pull_volume_parquet_forwards_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("local_pack.time.sleep", lambda s: None)

    class AlwaysFailsVol:
        def __init__(self):
            self.attempts = 0

        def read_file_into_fileobj(self, path, fileobj):
            self.attempts += 1
            raise OSError("boom")

    vol = AlwaysFailsVol()
    dst = tmp_path / "pulled.parquet"
    try:
        pull_volume_parquet(vol, "emb_0007.parquet", str(dst), retries=5)
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError after exhausting retries")
    assert vol.attempts == 5


def _seed_vol(files: dict, tensors: "dict[str, _torch.Tensor]"):
    vol = FakeVol()
    for name, t in tensors.items():
        buf = io.BytesIO(); _torch.save(t, buf)
        vol.files[f"embeddings/{name}"] = buf.getvalue()
    return vol


def test_run_shard_pack_uploads_and_pushes(tmp_path):
    names = [f"embeddings/e{i:04d}.pt" for i in range(4)]
    tensors = {os.path.basename(n): _bf16(3) for n in names}
    vol = _seed_vol({}, tensors)
    api = FakeApi()

    # shard 0 -> indices 0,1 ; shard_rows=2
    action = run_shard(vol, api, 0, names, shard_rows=2,
                       stage_dir=str(tmp_path / "stage"), em_repo="keypa/vision-adapter-embeddings", workers=1)
    assert action in ("pack", "push_from_vol")
    # volume got the parquet
    assert "emb_0000.parquet" in vol.uploaded
    # HF got pushed
    assert any(p == "data/emb_0000.parquet" for _, p, _ in api.calls)


def test_run_shard_skips_when_done(tmp_path):
    names = [f"embeddings/e{i:04d}.pt" for i in range(4)]
    vol = FakeVol()
    vol.files["shards/emb_0000.parquet"] = b"x"
    api = FakeApi()

    class ApiDone(FakeApi):
        def list_repo_files(self, repo_id, repo_type=None):
            return ["data/emb_0000.parquet"]

    action = run_shard(vol, ApiDone(), 0, names, shard_rows=2,
                       stage_dir=str(tmp_path / "stage"), em_repo="keypa/vision-adapter-embeddings", workers=1)
    assert action == "skip"
