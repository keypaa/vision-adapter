"""The packs can be published, so a FRESH session does not rebuild them.

A fresh Molab session has no data/ and no caches: without published packs,
every session rebuilds them (downloading ~175 GB of shards to do it), and
the rebuild is exactly where both of the pack bugs were. Published packs are
~3 GB and download in seconds.

The packs are derived from public corpus data plus the val manifest, so
publishing them next to the manifest is safe.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.pack_val as pv


class _Args:
    out_dir = "data"
    cache_dir = "data/cache"
    publish = True


def test_publish_uploads_the_pack_and_its_sidecar(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vision_adapter.data.pack import SCHEMA

    shard = tmp_path / "emb_0009.parquet"
    pq.write_table(pa.Table.from_pylist(
        [{"key": "k0", "n_vis": 364, "vis_bytes": b"x" * 8}], schema=SCHEMA),
        str(shard))

    uploaded = []

    class FakeApi:
        def upload_file(self, path_or_fileobj, path_in_repo, repo_id,
                        repo_type=None, commit_message=None):
            uploaded.append((Path(path_or_fileobj).name, path_in_repo, repo_id,
                             repo_type))

    import vision_adapter.data.stream as st

    orig_dl = st._download_shard_hf_transfer
    orig_api = None
    st._download_shard_hf_transfer = lambda sf, cache_dir=None: str(shard)
    import huggingface_hub

    orig_api = huggingface_hub.HfApi
    huggingface_hub.HfApi = FakeApi
    try:
        args = _Args()
        args.out_dir = str(tmp_path)
        pv._write_pack(args, "agentic",
                       [{"emb": "e0", "g": "agentic"}],
                       {"e0": ("emb_0009.parquet", 0, 364)},
                       str(tmp_path / "cache"))
    finally:
        st._download_shard_hf_transfer = orig_dl
        huggingface_hub.HfApi = orig_api

    names = [u[0] for u in uploaded]
    assert "val_pack_agentic.parquet" in names
    assert "val_pack_agentic.map.json" in names
    for _name, path_in_repo, repo_id, repo_type in uploaded:
        assert path_in_repo == _name, "the download must use the same flat name"
        assert repo_id == "keypa/vision-adapter-manifests"
        assert repo_type == "dataset"


def test_without_publish_nothing_is_uploaded(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vision_adapter.data.pack import SCHEMA

    shard = tmp_path / "emb_0009.parquet"
    pq.write_table(pa.Table.from_pylist(
        [{"key": "k0", "n_vis": 364, "vis_bytes": b"x" * 8}], schema=SCHEMA),
        str(shard))

    uploaded = []

    class FakeApi:
        def upload_file(self, *a, **k):
            uploaded.append(a)

    import vision_adapter.data.stream as st
    import huggingface_hub

    orig_dl = st._download_shard_hf_transfer
    orig_api = huggingface_hub.HfApi
    st._download_shard_hf_transfer = lambda sf, cache_dir=None: str(shard)
    huggingface_hub.HfApi = FakeApi
    try:
        args = _Args()
        args.out_dir = str(tmp_path)
        args.publish = False
        pv._write_pack(args, "agentic",
                       [{"emb": "e0", "g": "agentic"}],
                       {"e0": ("emb_0009.parquet", 0, 364)},
                       str(tmp_path / "cache"))
    finally:
        st._download_shard_hf_transfer = orig_dl
        huggingface_hub.HfApi = orig_api
    assert not uploaded
