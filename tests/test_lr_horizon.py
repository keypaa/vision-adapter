"""A bounded run must set its own LR horizon; it does not need one passed.

Regression (found on a real run, 2026-10-01): a bounded `--max-steps 200`
without `--lr-horizon` reached the training loop with `lr_horizon = None`,
because the assignment only existed on the resume branch. The cosine
schedule then died at the first step with
`TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'`
— after 100 steps of work and a saved checkpoint.

lr_at needs an end to decay toward. A bounded run *is* that end.

Run: python -m pytest tests/test_lr_horizon.py -q
"""
from vision_adapter.train import resolve_lr_horizon


def test_bounded_run_supplies_its_own_horizon():
    assert resolve_lr_horizon(steps=200, explicit=None) == 200


def test_explicit_horizon_wins_for_an_unbounded_run():
    assert resolve_lr_horizon(steps=None, explicit=5000) == 5000


def test_explicit_horizon_overrides_a_bounded_run():
    """A deliberate horizon is a different schedule; respect it."""
    assert resolve_lr_horizon(steps=200, explicit=5000) == 5000


def test_unbounded_without_horizon_is_rejected_before_the_loop():
    import pytest

    with pytest.raises(ValueError, match="lr_horizon"):
        resolve_lr_horizon(steps=None, explicit=None)
