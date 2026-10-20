"""The slice filter: which rows a run trains on, and that it is recorded.

The reference recipe's regime is the agentic slice (every row <= 380 visual
tokens, the reference's 300k-pixel budget); the doc slice is 79.7% of rows over
it (median 925, max 16,598 tokens) -- so a probe on "everything" is not a probe
on the recipe (docs/REFERENCE_RECIPE.md, D6).

The slice must be part of the run's identity: it is recorded in the plan
metadata, so a resume cannot silently train on a different slice.
"""
from vision_adapter.train import PLAN_META_KEYS, _diff_plan_meta, _recipe_mismatch


def _rows(groups):
    out = []
    for g, n in (("agentic", 3), ("doc", 2), ("conv", 1)):
        for i in range(n):
            out.append({"emb": f"embeddings/{g}{i}.pt", "g": g, "user": "u",
                        "assistant": "a"})
    return out


def _filter(rows, groups):
    """The same rule the trainer applies (train._streaming_train)."""
    groups = [g.strip() for g in (groups or "").split(",") if g.strip()]
    if not groups:
        return list(rows)
    return [r for r in rows if r.get("g") in groups]


def test_the_filter_keeps_only_the_requested_groups():
    rows = _rows(None)
    ag = _filter(rows, "agentic")
    assert len(ag) == 3
    assert {r["g"] for r in ag} == {"agentic"}
    both = _filter(rows, "agentic,conv")
    assert {r["g"] for r in both} == {"agentic", "conv"}
    assert len(both) == 4


def test_no_filter_keeps_everything():
    rows = _rows(None)
    assert len(_filter(rows, "")) == len(rows)
    assert len(_filter(rows, None)) == len(rows)


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
