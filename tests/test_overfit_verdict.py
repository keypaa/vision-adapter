"""The overfit test's verdict rule — the part that can be tested without a GPU.

The check is: can the projector move on a fixed batch? Two independent ways
to pass, because either clears the plumbing:
  * the best loss gets small (a memorised batch), or
  * the loss falls by a factor of five (learning, even if not yet memorised).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.overfit_test import verdict_for


def _curve(losses):
    return [{"step": i + 1, "loss": v, "gnorm": 0.2} for i, v in enumerate(losses)]


def test_a_flat_curve_is_a_failure():
    """Signature S4: the loss is pinned -> a bug, not a science problem."""
    v, first, best, last = verdict_for(_curve([1.1] * 10))
    assert v == "FAIL"
    assert first == best == last == 1.1


def test_a_memorised_batch_passes():
    v, first, best, last = verdict_for(_curve([1.1, 0.9, 0.5, 0.2, 0.05]))
    assert v == "PASS"
    assert best == 0.05


def test_a_fivefold_drop_passes_even_above_the_memorisation_bar():
    v, first, best, last = verdict_for(_curve([1.0, 0.8, 0.5, 0.3, 0.19]))
    assert v == "PASS"


def test_a_small_drop_is_a_failure():
    """1.1 -> 0.9 is not evidence that anything works."""
    v, *_ = verdict_for(_curve([1.1, 1.0, 0.9, 0.85]))
    assert v == "FAIL"


def test_an_empty_curve_is_a_failure():
    v, *_ = verdict_for([])
    assert v == "FAIL"
