"""W0b — the loader must serve every manifest question, and a checkpoint must
refuse a resume across a recipe change.

Two defects are pinned here:

1. The loader grouped plan rows by (row_group, position) and kept ONE row per
   position, so the manifest's extra questions about the same image (8,653
   images carry 2-54 of them) were silently never served — 45,393 of 117,600
   rows (39%) of the training signal, while `samples_seen` counted them.
2. A checkpoint did not record the recipe it was trained under, so a resume
   across a change of the row stream or of the collate budget would land on a
   different row (or different labels) and call it "continuous".

Hermetic: no HF, no GPU. Run:
    python -m pytest tests/test_serve_duplicates.py -q
"""

from vision_adapter.core import COLLATE_VERSION
from vision_adapter.data.stream import (
    STREAM_VERSION,
    group_rows_by_span,
    rows_in_span,
    span_is_skippable,
)
from vision_adapter.train import _recipe_mismatch


def _row(row_idx, emb="embeddings/x.pt", user="u", assistant="a"):
    return {"_row": row_idx, "emb": emb, "user": user, "assistant": assistant, "g": "t"}


def test_duplicates_are_all_kept_for_one_position():
    """Two questions about the same image must both survive the grouping."""
    rows = [_row(7, emb="embeddings/a.pt", assistant="q1"),
            _row(7, emb="embeddings/a.pt", assistant="q2")]
    want = group_rows_by_span(rows, g0=64)
    assert set(want) == {0}, "row 7 lives in row group 0"
    assert set(want[0]) == {7}
    assert [r["assistant"] for r in want[0][7]] == ["q1", "q2"], "both questions kept"


def test_grouping_is_deterministic_and_ordered():
    rows = [_row(0, assistant="a"), _row(64, assistant="b"), _row(0, assistant="c")]
    want = group_rows_by_span(rows, g0=64)
    assert sorted(want) == [0, 1], "row groups ascending"
    assert [r["assistant"] for r in want[0][0]] == ["a", "c"], "plan order preserved"
    again = group_rows_by_span(list(rows), g0=64)
    assert again == want, "grouping is a pure function of the input"


def test_served_row_count_equals_plan_row_count():
    """The whole point: the loader serves the plan, not a dedup of it."""
    rows = [_row(i) for i in range(10)]
    rows += [_row(3), _row(3), _row(7)]          # 3 extra questions
    want = group_rows_by_span(rows, g0=64)
    served = sum(len(v) for rg in want.values() for v in rg.values())
    assert served == len(rows) == 13


def test_positions_map_to_row_groups_by_g0():
    rows = [_row(0), _row(63), _row(64), _row(130)]
    want = group_rows_by_span(rows, g0=64)
    assert sorted(want) == [0, 1, 2]
    assert set(want[0]) == {0, 63}
    assert set(want[1]) == {0}
    assert set(want[2]) == {2}


def test_resume_refuses_a_stream_version_change():
    why = _recipe_mismatch({"stream_version": STREAM_VERSION - 1})
    assert why is not None and "row-stream" in why


def test_resume_refuses_a_collate_version_change():
    why = _recipe_mismatch({"collate_version": COLLATE_VERSION - 1})
    assert why is not None and "collate" in why


def test_resume_accepts_a_matching_recipe():
    assert _recipe_mismatch({"stream_version": STREAM_VERSION,
                             "collate_version": COLLATE_VERSION}) is None


def test_resume_accepts_a_legacy_checkpoint_without_versions():
    """Old checkpoints have no version keys: they are resumable (the loader
    change is the only difference and it is the caller's choice to resume)."""
    assert _recipe_mismatch({"manifest_sha256": "abc", "seed": 0}) is None


def test_rows_in_span_counts_every_question():
    rows = [_row(0), _row(0), _row(0), _row(5)]
    want = group_rows_by_span(rows, g0=64)
    assert rows_in_span(want[0]) == 4, "duplicates count as separate rows"


def test_span_is_skippable_only_when_every_row_is_before_the_offset():
    # 10 rows in the group, resume at row 10 -> the whole group is before it
    assert span_is_skippable(emitted=0, n_rows=10, start_pos=10) is True
    # the group straddles the offset -> must be fetched
    assert span_is_skippable(emitted=0, n_rows=11, start_pos=10) is False
    assert span_is_skippable(emitted=5, n_rows=5, start_pos=10) is True
    # a group that straddles the offset must be fetched
    assert span_is_skippable(emitted=7, n_rows=4, start_pos=10) is False
    # no resume -> nothing is skippable
    assert span_is_skippable(emitted=0, n_rows=5, start_pos=0) is False
