"""sample_size must not collapse when a run is unbounded.

Bug (audit 2026-09-30): `(max_steps or 5) * batch * 2` meant an unbounded run
built its data plan from 5 steps, i.e. 160 rows at batch 16 — and because
`build_epoch_plan` takes whole shards greedily, those 160 rows came from ONE
shard (n_vis median 66 vs 349 corpus-wide). Epoch replay then reshuffles
nothing, so the run memorises the same 160 images forever while the loss
falls. A healthy-looking run that is not a vision adapter at all.

`resolve_step_budget` already fixed the identical `or 5` pattern at the loop
boundary; this is the second call site it never reached.

Run: python -m pytest tests/test_sample_size.py -q
"""
import pytest

from vision_adapter.train import resolve_sample_size


def test_bounded_run_keeps_its_overrun_budget():
    assert resolve_sample_size(max_steps=100, batch_size=16, n_rows=117600) == 3200


def test_unbounded_run_covers_the_whole_manifest():
    """No step budget means no reason to subsample — use everything."""
    assert resolve_sample_size(max_steps=None, batch_size=16, n_rows=117600) == 117600


def test_never_exceeds_the_manifest():
    assert resolve_sample_size(max_steps=100000, batch_size=16, n_rows=500) == 500


def test_unbounded_tiny_manifest_is_fine():
    assert resolve_sample_size(max_steps=None, batch_size=8, n_rows=12) == 12


def test_batch_size_zero_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        resolve_sample_size(max_steps=10, batch_size=0, n_rows=100)
