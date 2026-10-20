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
    interleaved_positions,
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
    why = _recipe_mismatch({"stream_version": STREAM_VERSION,
                             "collate_version": COLLATE_VERSION - 1})
    assert why is not None and "collate" in why


def test_resume_accepts_a_matching_recipe():
    assert _recipe_mismatch({"stream_version": STREAM_VERSION,
                             "collate_version": COLLATE_VERSION}) is None


def test_resume_refuses_a_legacy_checkpoint_without_versions():
    """A checkpoint with no version keys predates BOTH changes, so it is v1 by
    definition: resuming it under the v2 loader would silently train on a
    different row stream with different labels. It must be refused (the
    operator can override with VISION_ADAPTER_ALLOW_LEGACY_RESUME=1)."""
    why = _recipe_mismatch({"manifest_sha256": "abc", "seed": 0})
    assert why is not None and "stream_version" in why


def test_resume_mismatch_check_survives_a_malformed_version(monkeypatch):
    """A malformed version must not silently disable the guard."""
    monkeypatch.delenv("VISION_ADAPTER_ALLOW_LEGACY_RESUME", raising=False)
    assert _recipe_mismatch({"stream_version": "not-a-number"}) is not None


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


def test_questions_of_one_image_are_interleaved():
    """Serving them consecutively would put 54 questions about ONE image in a
    batch, and the batch loss is a token-mean."""
    want = group_rows_by_span(
        [_row(0, assistant="q1"), _row(0, assistant="q2"), _row(1, assistant="r1")],
        g0=64)
    order = interleaved_positions(want[0])
    assert order == [(0, 0), (1, 0), (0, 1)], order
    served = [want[0][p][qi]["assistant"] for p, qi in order]
    assert served == ["q1", "r1", "q2"], served


# ------------------------------------------------------------- the loader ---
# Nothing drove EmbStreamDataset.__iter__ before: the emitted counter, the
# resume skip, the duplicates and the interleaving were all untested.


def _write_shard(path, n_rows=128, n_vis=4):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vision_adapter.data.pack import SCHEMA

    rows = [{"key": f"embeddings/k{i:04d}.pt", "n_vis": n_vis,
             "vis_bytes": b"\x00" * (n_vis * 4096 * 2)} for i in range(n_rows)]
    w = pq.ParquetWriter(path, SCHEMA, compression=None)
    for i in range(0, n_rows, 64):
        w.write_table(pa.Table.from_pylist(rows[i:i + 64], schema=SCHEMA))
    w.close()
    return path


def _plan_rows(n_positions=8, questions=2):
    out = []
    for p in range(n_positions):
        for qi in range(questions):
            out.append({"_row": p, "emb": f"embeddings/k{p:04d}.pt",
                        "user": f"u{p}q{qi}", "assistant": f"a{p}q{qi}", "g": "t"})
    return out


def test_loader_serves_every_plan_row_including_duplicates(tmp_path):
    from vision_adapter.data.stream import EmbStreamDataset

    shard = _write_shard(tmp_path / "emb_0000.parquet")
    plan_rows = _plan_rows(n_positions=8, questions=3)
    ds = EmbStreamDataset(
        plan={"data/emb_0000.parquet": plan_rows},
        stream_order=["data/emb_0000.parquet"],
        local_shards={"data/emb_0000.parquet": str(shard)},
    )
    got = list(ds)
    assert len(got) == len(plan_rows) == 24, len(got)
    texts = sorted(r["assistant"] for r in got)
    assert texts == sorted(r["assistant"] for r in plan_rows)
    # the questions of one image are spread, not consecutive
    served = [r["assistant"] for r in got]
    assert served[:3] == ["a0q0", "a1q0", "a2q0"], served[:3]


def test_loader_resume_skips_the_first_rows(tmp_path):
    from vision_adapter.data.stream import EmbStreamDataset, group_rows_by_span

    shard = _write_shard(tmp_path / "emb_0000.parquet")
    plan_rows = _plan_rows(n_positions=8, questions=2)
    ds = EmbStreamDataset(
        plan={"data/emb_0000.parquet": plan_rows},
        stream_order=["data/emb_0000.parquet"],
        start_pos=5,
        local_shards={"data/emb_0000.parquet": str(shard)},
    )
    got = list(ds)
    assert len(got) == len(plan_rows) - 5, len(got)
    # the expected served order: positions ascending, the questions of a
    # position spread round-robin
    want = group_rows_by_span(plan_rows, g0=64)
    expected = [want[0][p][qi]["assistant"]
                for p, qi in interleaved_positions(want[0])][5:]
    assert [r["assistant"] for r in got] == expected


def test_daemon_task_satisfies_the_drain_contract():
    """The prefetch result is consumed by drain_row_group_prefetch, which calls
    .result() -- a daemon task that only had .value() would silently break the
    adopt path (the prefetched bytes would be discarded and re-downloaded)."""
    from vision_adapter.data.stream import _DaemonTask, drain_row_group_prefetch

    t = _DaemonTask(lambda: b"hello")
    assert drain_row_group_prefetch(t) == b"hello"
    boom = _DaemonTask(lambda: 1 / 0)
    assert drain_row_group_prefetch(boom) is None      # errors are swallowed
    assert boom.done()


def test_the_current_recipe_versions_are_v3():
    """Both versions moved (the interleaving, the answer cap and the
    row-aligned crop), so the guard refuses every earlier recipe."""
    from vision_adapter.data.stream import STREAM_VERSION
    from vision_adapter.train import _recipe_mismatch

    assert STREAM_VERSION == 3
    assert COLLATE_VERSION == 3
    assert _recipe_mismatch({"stream_version": 2, "collate_version": 3}) is not None
    assert _recipe_mismatch({"stream_version": 3, "collate_version": 2}) is not None
    assert _recipe_mismatch({"stream_version": STREAM_VERSION,
                             "collate_version": COLLATE_VERSION}) is None
