"""The last-step val probe only fires if the val has already run once.

Observed on a real 200-step run (2026-10-01): the training loop finished in
1.5 min and then sat there, because the final-step probe had to stream 98
shards to materialise a 1,272-row val that `val_every=100000` had been
explicitly told not to fetch. Fifteen minutes of waiting after a completed
run reads as a hang.

Gating on "has the val already run" keeps the guarantee that matters — a long
run ends with a val loss on the final checkpoint, where the information is
freshest — while costing nothing on a run that never asked for the val.

Run: python -m pytest tests/test_val_final_step.py -q
"""
from vision_adapter.train import _val_due


def test_final_step_probes_when_the_val_has_been_running():
    """A long run must still end with a val loss on its last checkpoint."""
    assert _val_due(200, val_every=250, total_steps=200, has_run_before=True)


def test_final_step_does_not_probe_a_run_that_never_probed():
    """The regression: a smoke run asked for no val and paid for one anyway."""
    assert not _val_due(200, val_every=100000, total_steps=200, has_run_before=False)


def test_interval_probes_never_depend_on_prior_runs():
    """The mid-run cadence is unconditional — only the final step is gated."""
    assert _val_due(250, val_every=250, total_steps=2000, has_run_before=False)
    assert _val_due(500, val_every=250, total_steps=2000, has_run_before=False)


def test_a_val_that_only_fires_at_the_end_is_suppressed():
    """val_every >= steps means the final step IS the first probe."""
    assert not _val_due(200, val_every=200, total_steps=200, has_run_before=False)


def test_disabled_val_stays_disabled():
    assert not _val_due(250, val_every=0, total_steps=200, has_run_before=True)
    assert not _val_due(200, val_every=0, total_steps=200, has_run_before=True)
