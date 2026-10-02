"""The val probe must announce itself before it starts streaming.

Observed on a real run (2026-10-01): the training loop finished and the
terminal went silent for ~15 minutes while 98 shards streamed in. Nothing
distinguished "still working" from "dead" — the worst possible state for a
multi-hour run you are watching remotely.

Two lines fix it: one when the probe starts (with the row and shard count,
so the wait is quantified), one when it lands.

Run: python -m pytest tests/test_val_progress.py -q
"""
from vision_adapter.train import _val_progress_lines


def test_start_line_names_the_work_to_come():
    start, _done = _val_progress_lines(
        step=250, n_rows=1272, n_shards=98, elapsed_s=3.0
    )
    assert "VAL" in start
    assert "1272" in start, "the operator must know how many rows are coming"
    assert "98" in start, "and how many shards will be touched"
    assert "3.0" in start, "and how long the last one took"


def test_done_line_carries_the_loss():
    _start, done = _val_progress_lines(
        step=250, n_rows=1272, n_shards=98, elapsed_s=3.0
    )
    assert "3.0" in done


def test_first_probe_says_it_is_the_first():
    """The first probe is the expensive one — say so explicitly."""
    start, _ = _val_progress_lines(
        step=250, n_rows=1272, n_shards=98, elapsed_s=0.0, first=True
    )
    assert "first" in start.lower()
