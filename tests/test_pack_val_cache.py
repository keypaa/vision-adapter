"""The pack step must not leave ~175 GB of shards behind.

pyarrow's parquet API needs a LOCAL file, so pack_val downloads a whole
shard (0.4-60 GB) for each shard that holds val rows -- ~50 of them. Into a
temp dir that is thrown away afterwards, that is ~175 GB of downloads AND
~175 GB of disk that the trainer then has to re-download and re-store.

So the shards go into the TRAINER's cache dir (data/cache), and the LRU
trims it after every shard: the peak stays inside the disk budget, and the
shards that survive are the ones the trainer reuses instead of re-fetching.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.pack_val as pv


class _Args:
    manifest = None
    train_manifest = None
    key_index = None
    cache_dir = "data/cache"
    out_dir = "data"
    n = 10
    seed = 0
    dry_run = False


def test_the_pack_uses_the_trainers_cache_dir(monkeypatch, tmp_path):
    seen = {}

    def fake_write_pack(args, group, rows, index, cache_dir):
        seen["cache_dir"] = cache_dir
        seen["group"] = group
        return 0

    monkeypatch.setattr(pv, "_write_pack", fake_write_pack)
    monkeypatch.setattr(pv, "key_index", lambda a: {
        "e1": ("data/emb_0003.parquet", 0, 364),
        "e2": ("data/emb_0004.parquet", 1, 352)})
    monkeypatch.setattr(pv, "quotas_from_corpus",
                        lambda rows, index, n: ({"agentic": 10}, {1: 10}))
    monkeypatch.setattr(pv, "pick_per_group",
                        lambda rows, index, gq, bs, seed=0: {"agentic": rows})
    # the manifests are fetched from HF inside pack(); give it local files
    mp = tmp_path / "val.jsonl"
    mp.write_text("")
    tp = tmp_path / "train.jsonl"
    tp.write_text("")
    args = _Args()
    args.manifest = str(mp)
    args.train_manifest = str(tp)
    args.cache_dir = str(tmp_path / "cache")
    monkeypatch.setattr("vision_adapter.train._val_rows_from_file",
                        lambda p: [{"emb": "e1", "g": "agentic"},
                                   {"emb": "e2", "g": "agentic"}]
                        if "val" in str(p) else
                        [{"emb": "t1", "g": "agentic"}])
    assert pv.pack(args) == 0
    assert seen["cache_dir"] == args.cache_dir, \
        "the shards must land in the trainer's cache, not a temp dir"
    assert "valpack_" not in seen["cache_dir"]
