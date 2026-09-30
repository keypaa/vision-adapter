"""max_steps=None must mean "run until stopped", not "run 5 steps".

The streaming loop had `steps = max_steps or 5`, so modal_train.py:1318's
`run_train(..., max_steps=None)` — written in the belief that None meant "no
limit" — silently trained 5 steps and exited 0. A 5-step run looks like a
successful run.

None is unbounded. That needs a decision the current code cannot make: the LR
schedule is a cosine over max_steps, so with no bound there is no decay. The
bound becomes explicit and mandatory, named for what it is.

Run: python -m pytest tests/test_max_steps.py -q
"""
import pytest

from vision_adapter.train import resolve_step_budget


def test_explicit_steps_pass_through():
    assert resolve_step_budget(2000) == 2000


def test_none_means_unbounded():
    assert resolve_step_budget(None, lr_horizon=5000) is None


def test_zero_and_negative_are_rejected():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_steps"):
            resolve_step_budget(bad)


def test_unbounded_run_needs_an_explicit_lr_horizon():
    """Without a decay horizon a cosine LR never decays — silently."""
    with pytest.raises(ValueError, match="lr_horizon"):
        resolve_step_budget(None, lr_horizon=None)


def test_unbounded_run_uses_the_lr_horizon_for_the_schedule():
    assert resolve_step_budget(None, lr_horizon=5000) is None


def test_bounded_run_needs_no_horizon():
    """A known end already defines the schedule."""
    assert resolve_step_budget(1000, lr_horizon=None) == 1000
