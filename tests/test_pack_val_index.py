"""The pack step must build the key index when the box does not have one yet.

pack_val.py locates the val rows' (shard, row, n_vis) through the key index.
On a fresh box that file does not exist — the trainer builds one at startup,
but the pack step runs BEFORE the trainer. So the pack step has to be able
to build it itself, or the runbook's "pack the held-out set" step dies with
"no key index found" on every fresh host.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pack_val import key_index


class _Args:
    key_index = None
    cache_dir = "data/cache"


def test_a_missing_index_is_built(monkeypatch, tmp_path):
    built = {}

    def fake_build(order, cache_dir=None, rebuild=False):
        built["order"] = order
        built["cache_dir"] = cache_dir
        return {"embeddings/x.pt": ("data/emb_0000.parquet", 0, 364)}

    monkeypatch.setattr("vision_adapter.data.stream.build_key_index", fake_build)
    monkeypatch.setattr("vision_adapter.data.stream.list_shards",
                        lambda *a, **k: ["data/emb_0000.parquet", "data/emb_0001.parquet"])
    args = _Args()
    args.cache_dir = str(tmp_path / "cache")
    idx = key_index(args)
    assert idx, "the index must not be empty"
    assert built["order"] == ["data/emb_0000.parquet", "data/emb_0001.parquet"]
    assert built["cache_dir"] == args.cache_dir


def test_an_existing_index_is_used_as_is(monkeypatch, tmp_path):
    import json

    p = tmp_path / "key_index_cache.json"
    p.write_text(json.dumps({"version": 3, "keys": {
        "embeddings/x.pt": {"shard": "data/emb_0009.parquet", "row": 3, "n_vis": 364}}}))
    monkeypatch.setattr("vision_adapter.data.stream.build_key_index",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build")))
    args = _Args()
    args.key_index = str(p)
    idx = key_index(args)
    assert idx["embeddings/x.pt"][2] == 364
