"""Key-index cache must be namespaced by shard set.

Regression: eval over emb_0000/0001 loaded the training index (13 shards)
from the shared cache file and resolved 0 rows (fail-closed assert fired).
"""
import json

from vision_adapter.data.stream import _cache_key_index_path, load_key_index, save_key_index


def test_cache_path_namespaced_by_shard_set():
    a = _cache_key_index_path("cache", ["data/emb_0001.parquet", "data/emb_0000.parquet"])
    b = _cache_key_index_path("cache", ["data/emb_0000.parquet", "data/emb_0001.parquet"])
    c = _cache_key_index_path("cache", ["data/emb_0002.parquet"])
    assert a == b  # order-insensitive
    assert a != c  # different set -> different file
    assert a.endswith(".json")


def test_cache_path_legacy_without_order():
    assert _cache_key_index_path("cache").endswith("key_index_cache.json")


def test_save_load_roundtrip_namespaced(tmp_path):
    from vision_adapter.data.stream import _cache_key_index_path as p

    path = p(str(tmp_path), ["data/emb_0000.parquet"])
    save_key_index({"k": ("data/emb_0000.parquet", 3, 12)}, path)
    index, ok = load_key_index(path)
    assert ok and index == {"k": ("data/emb_0000.parquet", 3, 12)}
    assert json.load(open(path))["version"] == 3
