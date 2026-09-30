"""A run must never quietly train on the synthetic geometry stand-in.

The stand-in inverts orientation: measured 2026-09-30 over the 117600-row
manifest, grid_for_nvis differs from the true MoonViT grid on 38.1% of rows
and INVERTS portrait/landscape on 18.3% of them (aspect error up to x89).
A 20-hour run on that geometry would look healthy and be wrong.

So the run refuses to start unmeasured, unless the caller explicitly opts in
— a baseline on the old geometry is a legitimate experiment, but it has to be
asked for by name, not arrived at by accident.

Run: python -m pytest tests/test_grid_source_guard.py -q
"""
import pytest

from vision_adapter.manifest import grid_source_for
from vision_adapter.train import geometry_guard


def test_full_coverage_is_measured():
    rows = [{"emb": "a", "grid_thw": [1, 4, 8]}, {"emb": "b", "grid_thw": [1, 2, 4]}]
    assert grid_source_for(*_counts(rows)) == "measured"


def test_no_coverage_is_synthetic():
    rows = [{"emb": "a"}, {"emb": "b"}]
    assert grid_source_for(*_counts(rows)) == "synthetic"


def test_half_coverage_is_partial_never_measured():
    """Claiming full coverage on half the data is the lie this prevents."""
    rows = [{"emb": "a", "grid_thw": [1, 4, 8]}, {"emb": "b"}]
    assert grid_source_for(*_counts(rows)) == "partial"


def _counts(rows):
    have = sum(1 for r in rows if r.get("grid_thw"))
    return have, len(rows) - have


def test_measured_geometry_passes_the_guard():
    geometry_guard("measured", allow_synthetic=False)      # must not raise


def test_synthetic_geometry_blocks_the_run():
    with pytest.raises(RuntimeError, match="synthetic"):
        geometry_guard("synthetic", allow_synthetic=False)


def test_partial_geometry_blocks_the_run_too():
    with pytest.raises(RuntimeError, match="partial"):
        geometry_guard("partial", allow_synthetic=False)


def test_explicit_opt_in_allows_a_synthetic_baseline():
    """Asking for the old geometry by name is a legitimate experiment."""
    geometry_guard("synthetic", allow_synthetic=True)
    geometry_guard("partial", allow_synthetic=True)


def test_the_error_names_the_way_out():
    """The failure must tell the operator how to proceed, not just refuse."""
    with pytest.raises(RuntimeError) as e:
        geometry_guard("synthetic", allow_synthetic=False)
    msg = str(e.value)
    assert "train_manifest_grids.jsonl" in msg
    assert "allow_synthetic" in msg
