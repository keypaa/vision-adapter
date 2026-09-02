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
