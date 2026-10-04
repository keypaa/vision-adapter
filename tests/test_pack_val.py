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

def test_pick_never_exceeds_a_quota():
    val = [{"emb": f"v{i}", "g": "agentic"} for i in range(500)]
    index = _index([(r["emb"], 374) for r in val])
    chosen, _ = pv.pick_rows(val, index, {"agentic": 100}, {1: 100}, seed=0)
    assert len(chosen) == 100


def test_pick_has_no_duplicate_keys():
    val = [{"emb": f"v{i}", "g": "doc"} for i in range(300)]
    index = _index([(r["emb"], 200) for r in val])
    chosen, _ = pv.pick_rows(val, index, {"doc": 200}, {1: 200}, seed=0)
    assert len({r["emb"] for r in chosen}) == len(chosen)


def test_a_rare_bucket_is_not_starved_by_an_abundant_one():
    """11 317 rows sit in 0-100 and 92 801 in 101-500. If the filler took from
    the plentiful cell first, the rare cell would never be reached."""
    small = [{"emb": f"s{i}", "g": "agentic"} for i in range(5)]
    big = [{"emb": f"b{i}", "g": "agentic"} for i in range(400)]
    val = small + big
    index = _index([(r["emb"], 50 if r["emb"].startswith("s") else 374) for r in val])

    chosen, _ = pv.pick_rows(val, index, {"agentic": 10}, {0: 5, 1: 5}, seed=0)
    got = Counter_of = [pv.bucket_of(int(index[r["emb"]][2])) for r in chosen]
    assert Counter_of.count(0) == 5, "all five small rows must be taken"
    assert Counter_of.count(1) == 5


def test_deficit_is_reported_when_the_val_split_cannot_fill_the_quota():
    val = [{"emb": "only1", "g": "agentic"}]
    index = _index([("only1", 374)])
    chosen, deficit = pv.pick_rows(val, index, {"agentic": 500}, {1: 500}, seed=0)
    assert len(chosen) == 1
    assert deficit > 0, "a shortfall must be visible, not silently shrink the pack"


def test_rows_without_grid_or_index_are_skipped():
    val = [{"emb": "ghost", "g": "agentic"}, {"emb": "real", "g": "agentic"}]
    index = _index([("real", 374)])
    chosen, _ = pv.pick_rows(val, index, {"agentic": 10}, {1: 10}, seed=0)
    assert [r["emb"] for r in chosen] == ["real"]


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