"""grid_thw rides in the manifest itself — no sidecar file, no config knob.

Refactor 2026-09-30. The sidecar existed because I was reluctant to touch the
manifest, which produced a second index of the same data plus a config field,
a cache file, and a load path — four things to keep in sync for 20 bytes per
row. The manifest already carries {emb, user, assistant, g} and IS the plan
that training iterates, so the measured grid belongs there. Cost: ~20 bytes
per row, +3% on an 82 MB manifest.

The recovery itself is unchanged and still verified: the grid is a
deterministic function of the image dims under the preprocess contract
(400/400 live rows matched the stored n_vis), so the backfill is a one-time
rewrite of the manifest from the image corpus.

Run: python -m pytest tests/test_grid_sidecar.py -q
"""
import json

from vision_adapter.manifest import (
    grid_thw_for_row,
    manifest_has_grids,
)


def _row(emb, g="doc", grid=None):
    r = {"emb": emb, "user": "u", "assistant": "a", "g": g}
    if grid is not None:
        r["grid_thw"] = grid
    return r


def test_row_with_a_grid_is_measured():
    assert grid_thw_for_row(_row("e", grid=[1, 56, 26])) == [1, 56, 26]


def test_row_without_a_grid_is_synthetic():
    """A manifest not yet backfilled must still train, on the stand-in."""
    assert grid_thw_for_row(_row("e")) is None


def test_manifest_grid_coverage_is_reportable():
    """The backfill needs to know how far it got, per group."""
    rows = [_row("a", "agentic", [1, 4, 8]), _row("b", "agentic"),
            _row("c", "doc", [1, 2, 4])]
    have, missing = manifest_has_grids(rows)
    assert have == 2 and missing == 1


def test_manifest_without_grids_reports_zero_coverage():
    rows = [_row("a"), _row("b")]
    have, missing = manifest_has_grids(rows)
    assert (have, missing) == (0, 2)


def test_grid_survives_a_manifest_roundtrip(tmp_path):
    """A rewrite must not drop grid_thw — that is the whole point."""
    from vision_adapter.manifest import load_manifest, write_manifest_with_header

    rows = [_row("e1", "agentic", [1, 56, 26]), _row("e2", "doc", [1, 52, 28])]
    p = write_manifest_with_header(tmp_path / "m.jsonl", rows)
    back, _header = load_manifest(p)
    assert [r["grid_thw"] for r in back] == [[1, 56, 26], [1, 52, 28]]


def test_legacy_manifest_still_loads_without_grids(tmp_path):
    """Backwards compatibility: old manifests have no grid_thw at all."""
    from vision_adapter.manifest import load_manifest

    p = tmp_path / "legacy.jsonl"
    p.write_text(
        json.dumps({"emb": "e1", "user": "u", "assistant": "a", "g": "doc"}) + "\n"
    )
    rows, _ = load_manifest(p)
    assert grid_thw_for_row(rows[0]) is None


def test_dataset_no_longer_takes_a_sidecar_path():
    """The config knob is gone; rows carry their own geometry."""
    import inspect

    from vision_adapter.data.stream import EmbStreamDataset

    sig = inspect.signature(EmbStreamDataset.__init__)
    assert "grid_sidecar" not in sig.parameters
