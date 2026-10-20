"""The slice filter, applied to the training rows AND the held-out rows.

The reference recipe's regime is the agentic slice (every row <= 380 visual
tokens, the reference's 300k-pixel budget); the doc slice is 79.7% of rows over
it (median 925, max 16,598 tokens) -- so a probe on "everything" is not a probe
on the recipe (docs/REFERENCE_RECIPE.md, D6).

The held-out probe must measure the SAME slice: the val split is 85% agentic
(1,076 of 1,272 rows) but also carries 196 doc/conv rows. On an agentic-only
run those 196 rows are rows the model never trained on -- a constant offset
that can mask the very signal the probe is read for.
"""
from vision_adapter.train import (
    PLAN_META_KEYS,
    _diff_plan_meta,
    _recipe_mismatch,
    filter_by_groups,
)


def _rows(groups):
    out = []
    for g, n in (("agentic", 3), ("doc", 2), ("conv", 1)):
        for i in range(n):
            out.append({"emb": f"embeddings/{g}{i}.pt", "g": g, "user": "u",
                        "assistant": "a"})
    return out


def test_the_filter_keeps_only_the_requested_groups():
    rows = _rows(None)
    ag = filter_by_groups(rows, "agentic")
    assert len(ag) == 3
    assert {r["g"] for r in ag} == {"agentic"}
    both = filter_by_groups(rows, "agentic,conv")
    assert {r["g"] for r in both} == {"agentic", "conv"}
    assert len(both) == 4


def test_no_filter_keeps_everything():
    rows = _rows(None)
    assert len(filter_by_groups(rows, "")) == len(rows)
    assert len(filter_by_groups(rows, None)) == len(rows)
    assert len(filter_by_groups(rows, [])) == len(rows)


def test_the_same_rule_serves_the_train_and_the_val_rows():
    """One rule, so the held-out probe cannot measure another slice."""
    rows = _rows(None)
    val = [r for r in rows if r["g"] == "agentic"]
    assert filter_by_groups(val, "agentic") == val
    # a val row from another slice is dropped, not silently kept
    mixed = rows[:2]
    assert filter_by_groups(mixed, "agentic") == [r for r in mixed if r["g"] == "agentic"]


def test_the_slice_is_part_of_the_plan_identity():
    """A resume on a different slice must be flagged."""
    assert "groups" in PLAN_META_KEYS
    cur = {"groups": "agentic", "lr_schedule": "constant"}
    ckpt = {"groups": "", "lr_schedule": "cosine"}
    diffs = _diff_plan_meta(cur, ckpt)
    assert "groups" in diffs and "lr_schedule" in diffs


def test_the_slice_alone_does_not_block_a_resume():
    """The slice is a plan-metadata warning, not a recipe change: the rows and
    their order are unchanged, so a resume is still valid."""
    assert _recipe_mismatch({"stream_version": 3, "collate_version": 3,
                              "groups": "agentic"}) is None
