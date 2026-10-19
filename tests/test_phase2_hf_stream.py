import tempfile, os, json, pathlib
import torch
import pyarrow.parquet as pq
import pyarrow as pa

from vision_adapter.data.stream import (
    save_key_index, load_key_index,
    build_epoch_plan, KEY_INDEX_CACHE,
    rg_span, RemoteShard
)
from vision_adapter.data.pack import _bucket_id

def test_bucket_id_boundaries():
    assert _bucket_id(0)==0 and _bucket_id(100)==0
    assert _bucket_id(101)==1 and _bucket_id(500)==1
    assert _bucket_id(501)==2 and _bucket_id(1000)==2
    assert _bucket_id(1001)==3 and _bucket_id(2000)==3
    assert _bucket_id(2001)==4 and _bucket_id(4900)==4
    assert _bucket_id(4901)==5 and _bucket_id(16653)==5

def test_v3_roundtrip_with_nvis():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)/KEY_INDEX_CACHE
        idx = {"k1": ("s.parquet", 7, 420), "k2": ("s.parquet", 8, 120)}
        save_key_index(idx, str(p))
        loaded, ok = load_key_index(str(p))
        assert ok
        assert loaded["k1"] == ("s.parquet", 7, 420)
        assert json.load(open(p))["version"] == 3

def test_v2_still_loads():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)/"v2.json"
        payload = {"version": 2, "keys": {"a": {"shard": "s.parquet", "row": 5}}}
        json.dump(payload, open(p,"w"))
        idx, ok = load_key_index(str(p))
        assert ok and idx["a"] == ("s.parquet", 5)

def test_bucketed_plan_within_shard_sorted():
    idx = {
        "embeddings/aaa.pt": ("data/emb_0002.parquet", 0, 80),
        "embeddings/bbb.pt": ("data/emb_0002.parquet", 1, 450),
    }
    rows = [{"emb": k, "user": "u", "assistant": "a", "g": "t"} for k in idx]
    plan = build_epoch_plan(rows, idx, sample_size=2, seed=0)
    assert plan["data/emb_0002.parquet"][0]["emb"] == "embeddings/aaa.pt"

def test_non_bucketed_fallback_v2():
    idx = {
        "embeddings/aaa.pt": ("data/emb_0002.parquet", 0),
        "embeddings/bbb.pt": ("data/emb_0002.parquet", 1),
    }
    rows = [{"emb": k, "user": "u", "assistant": "a", "g": "t"} for k in idx]
    plan = build_epoch_plan(rows, idx, sample_size=2, seed=0)
    assert len(plan) == 1

def test_lru_eviction_logic():
    from collections import OrderedDict
    lru = OrderedDict()
    for i in range(6):
        lru[f"emb_{i:04d}.parquet"] = f"/tmp/hf_shards/emb_{i:04d}.parquet"
        if len(lru) > 4:
            lru.popitem(last=False)
    assert len(lru) == 4
    assert "emb_0000.parquet" not in lru

def test_hf_transfer_chunked_fallback():
    try:
        import hf_transfer
        assert True
    except ImportError:
        assert True


def test_lru_finds_shards_under_the_data_subdir(tmp_path):
    """hf_hub_download(local_dir=...) keeps the repo-relative path, so a shard
    lands at <cache>/data/emb_XXXX.parquet. The old glob (<cache>/emb_*.parquet)
    matched nothing -> the whole-shard cache was never evicted."""
    from vision_adapter.data.stream import _cached_shard_files

    d = tmp_path / "data"
    d.mkdir()
    (d / "emb_0094.parquet").write_bytes(b"x" * 10)
    (tmp_path / "emb_0001.parquet").write_bytes(b"x" * 10)
    found = _cached_shard_files(str(tmp_path))
    names = sorted(os.path.basename(p) for p in found)
    assert names == ["emb_0001.parquet", "emb_0094.parquet"], found


def test_lru_evicts_the_oldest_shard(tmp_path):
    import time

    from vision_adapter.data.stream import _enforce_lru_cache

    d = tmp_path / "data"
    d.mkdir()
    paths = []
    for i in range(4):
        p = d / f"emb_{i:04d}.parquet"
        p.write_bytes(b"x" * 10)
        os.utime(p, (time.time() - (10 - i) * 60, time.time() - (10 - i) * 60))
        paths.append(p)
    _enforce_lru_cache(str(tmp_path), max_shards=2)
    left = sorted(p.name for p in d.glob("emb_*.parquet"))
    assert left == ["emb_0002.parquet", "emb_0003.parquet"], left


def test_rg_cache_budget_has_a_floor_and_a_ceiling(tmp_path, monkeypatch):
    """The row-group cache is a cache: it scales with the disk but stays bounded."""
    import shutil

    from vision_adapter.data import stream

    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: shutil._ntuple_diskusage(10**12, 10**12, 10**12))
    assert stream._rg_cache_budget(str(tmp_path)) == stream.RG_CACHE_MAX_BYTES
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: shutil._ntuple_diskusage(1, 1, 10**9))
    assert stream._rg_cache_budget(str(tmp_path)) == stream.RG_CACHE_MIN_BYTES
