"""pack_val.py must produce a val set that is disjoint BY KEY and representative.

The disjointness is the whole point: run_train's val plan drops 1272 rows to
77 over 21 shards, and if the pack did not exclude the train keys the val_loss
would be measuring memorisation. The representative part matters too — shards
are sorted by n_vis, so a naive head-of-list pack would be all 0-100 bucket.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "pack_val.py"

sys.path.insert(0, str(SCRIPT.parent))
_spec = importlib.util.spec_from_file_location("pack_val", SCRIPT)
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)


def _index(pairs):
    """emb -> (shard, row, n_vis)"""
    return {e: (f"data/emb_{i:04d}.parquet", i, nv) for i, (e, nv) in enumerate(pairs)}


# ---------------------------------------------------------------- buckets

def test_bucket_edges_match_the_dataset_card():
    assert pv.bucket_of(0) == 0
    assert pv.bucket_of(100) == 0
    assert pv.bucket_of(101) == 1
    assert pv.bucket_of(500) == 1
    assert pv.bucket_of(1000) == 2
    assert pv.bucket_of(4900) == 4
    assert pv.bucket_of(4901) == 5
    assert pv.bucket_of(16653) == 5, "the max in the corpus must land in 4901+"


def test_bucket_name_is_readable():
    assert pv.bucket_name(374) == "101-500"
    assert pv.bucket_name(4901) == "4901+", "the open-ended bucket must not print 1e+09"


# ---------------------------------------------------------------- quotas

def test_quotas_track_the_train_corpus_not_the_val_split():
    """Val is 1272 rows so its own marginals are noisy; quotas come from train."""
    train = [{"emb": f"t{i}", "g": "agentic"} for i in range(450)]
    train += [{"emb": f"d{i}", "g": "doc"} for i in range(450)]
    train += [{"emb": f"c{i}", "g": "conv"} for i in range(100)]
    index = {r["emb"]: ("data/emb_0002.parquet", i, 374) for i, r in enumerate(train)}

    g_q, b_q = pv.quotas_from_corpus(train, index, n_total=1000)

    assert g_q["agentic"] == 450
    assert g_q["doc"] == 450
    assert g_q["conv"] == 100
    assert b_q[1] == 1000, "every row is 101-500 so that bucket takes the whole quota"


def test_quotas_are_proportional_across_buckets():
    train = [{"emb": f"a{i}", "g": "agentic"} for i in range(600)]   # small
    train += [{"emb": f"b{i}", "g": "doc"} for i in range(400)]      # large
    index = {}
    for i, r in enumerate(train):
        nv = 50 if r["emb"].startswith("a") else 5000
        index[r["emb"]] = ("data/emb_0002.parquet", i, nv)

    _, b_q = pv.quotas_from_corpus(train, index, n_total=1000)
    assert b_q[0] == 600
    assert b_q[5] == 400


def test_rows_without_an_index_entry_are_not_counted():
    """A missing embedding must not distort the corpus proportions."""
    train = [{"emb": f"t{i}", "g": "agentic"} for i in range(500)]
    train += [{"emb": "ghost", "g": "doc"}]          # not in the index
    index = {f"t{i}": ("data/emb_0002.parquet", i, 374) for i in range(500)}

    g_q, _ = pv.quotas_from_corpus(train, index, n_total=1000)
    assert "doc" not in g_q, "a group with no indexable row gets no quota"
    assert g_q["agentic"] == 1000, "the ghost row must not dilute the real mix"


def test_a_manifest_with_no_indexable_rows_fails_loudly():
    """Rather than pack 0 rows silently, say the key index is stale."""
    with pytest.raises(SystemExit, match="no indexable train rows"):
        pv.quotas_from_corpus([{"emb": "ghost", "g": "doc"}], {}, n_total=1000)


# ---------------------------------------------------------------- picking


def _share(*vals):
    """Bucket shares by bucket index, zeros for the buckets not given."""
    tot = sum(vals) or 1
    return {i: v / tot for i, v in enumerate(vals)}


def _all_in(bucket, n):
    """Every row lands in `bucket` — the common case in these fixtures."""
    return _share(*(n if i == bucket else 0 for i in range(6)))


def test_each_group_gets_its_own_quota():
    val = ([{"emb": f"a{i}", "g": "agentic"} for i in range(900)]
           + [{"emb": f"d{i}", "g": "doc"} for i in range(400)])
    index = _index([(r["emb"], 374) for r in val])

    packs = pv.pick_per_group(val, index, {"agentic": 750, "doc": 250},
                              _all_in(1, 1000), seed=0)
    assert len(packs["agentic"]) == 750
    assert len(packs["doc"]) == 250


def test_a_group_shorter_than_its_quota_is_not_topped_up_from_another():
    """The live shortfall: the val split holds ~60 doc rows against a quota of
    270. Stealing agentic rows to fill doc would make both numbers fiction."""
    val = ([{"emb": f"a{i}", "g": "agentic"} for i in range(900)]
           + [{"emb": f"d{i}", "g": "doc"} for i in range(60)])
    index = _index([(r["emb"], 374) for r in val])

    packs = pv.pick_per_group(val, index, {"agentic": 750, "doc": 270},
                              _all_in(1, 1000), seed=0)
    assert len(packs["agentic"]) == 750, "agentic keeps its own quota"
    assert len(packs["doc"]) == 60, "doc takes what exists, no more"


def test_a_group_with_no_rows_yields_an_empty_pack_not_a_crash():
    index = _index([("a", 374)])
    packs = pv.pick_per_group([{"emb": "a", "g": "agentic"}], index,
                              {"agentic": 10, "doc": 5}, _all_in(1, 1000), seed=0)
    assert packs["doc"] == [], "a missing group must be empty, not fatal"


def test_repeated_emb_keys_are_deduplicated():
    """The live val manifest repeats emb keys. Two rows sharing an embedding
    are the same sample twice — the loss would look steadier than it is."""
    val = [{"emb": "dup", "g": "agentic"}, {"emb": "dup", "g": "agentic"},
           {"emb": "other", "g": "agentic"}]
    index = _index([("dup", 374), ("other", 374)])
    packs = pv.pick_per_group(val, index, {"agentic": 10}, _all_in(1, 1000), seed=0)
    embs = [r["emb"] for r in packs["agentic"]]
    assert len(embs) == len(set(embs)), "a duplicated key must not appear twice"


def test_each_pack_keeps_the_corpus_bucket_mix():
    """A doc pack must still span small and large images."""
    small = [{"emb": f"s{i}", "g": "doc"} for i in range(40)]
    big = [{"emb": f"b{i}", "g": "doc"} for i in range(400)]
    val = small + big
    index = _index([(r["emb"], 50 if r["emb"].startswith("s") else 374)
                    for r in val])

    packs = pv.pick_per_group(val, index, {"doc": 100}, _share(0.1, 0.9), seed=0)
    got = [pv.bucket_of(int(index[r["emb"]][2])) for r in packs["doc"]]
    assert got.count(0) == 10, "10% of the pack must be small"
    assert got.count(1) == 90


def test_a_bucket_absent_from_the_val_split_does_not_break_the_pack():
    """The val split holds 77 usable rows; a 6-bucket quota cannot be met."""
    val = [{"emb": f"v{i}", "g": "agentic"} for i in range(100)]
    index = _index([(r["emb"], 374) for r in val])
    packs = pv.pick_per_group(val, index, {"agentic": 100},
                              _all_in(1, 1000), seed=0)
    assert len(packs["agentic"]) == 100, "the pack fills from the bucket it has"


def test_rows_without_an_index_entry_are_skipped():
    val = [{"emb": "ghost", "g": "agentic"}, {"emb": "real", "g": "agentic"}]
    index = _index([("real", 374)])
    packs = pv.pick_per_group(val, index, {"agentic": 10}, _all_in(1, 1000), seed=0)
    assert [r["emb"] for r in packs["agentic"]] == ["real"]


def test_rows_come_out_in_bucket_order():
    """Batches must stay size-homogeneous; sorted n_vis is how the plan does it."""
    import random

    rng = random.Random(0)
    val = [{"emb": f"v{i}", "g": "agentic"} for i in range(100)]
    index = {f"v{i}": (f"data/emb_0002.parquet", i, rng.randint(50, 4000))
             for i in range(100)}
    packs = pv.pick_per_group(val, index, {"agentic": 100}, _all_in(1, 1000), seed=0)
    sizes = [index[r["emb"]][2] for r in packs["agentic"]]
    assert sizes == sorted(sizes)


def test_the_same_seed_gives_the_same_pack():
    val = [{"emb": f"v{i}", "g": "agentic"} for i in range(200)]
    index = _index([(r["emb"], 50 if i % 2 else 374) for i, r in enumerate(val)])
    a = pv.pick_per_group(val, index, {"agentic": 100}, _share(1, 1), seed=3)
    b = pv.pick_per_group(val, index, {"agentic": 100}, _share(1, 1), seed=3)
    assert [r["emb"] for r in a["agentic"]] == [r["emb"] for r in b["agentic"]]


# ---------------------------------------------------------------- CLI shape

def test_dry_run_is_opt_in_not_the_default():
    src = SCRIPT.read_text()
    assert '"--dry-run"' in src
    assert 'action="store_true"' in src
    assert "if args.dry_run:" in src, "the dry run must short-circuit before any fetch"


def test_disjoint_is_checked_before_writing():
    """The assert has to sit before pack_rows, not after."""
    src = SCRIPT.read_text()
    assert "overlaps train" in src
    assert src.index("overlaps train") < src.index("pack_rows(out_rows"), \
        "the disjoint check must run before anything is written"


def test_a_source_key_to_emb_map_is_written():
    """The pack's `key` is the SOURCE key; the manifest speaks in `emb`. The map
    is the only thing that joins them, so it must exist."""
    src = SCRIPT.read_text()
    assert ".map.json" in src, \
        "we must be able to check later what the val actually contained"
    assert "source_to_emb[k] = r[\"emb\"]" in src, \
        "the map must be built as rows are packed, not reconstructed later"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))