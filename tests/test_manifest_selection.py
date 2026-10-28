"""The trainer prefers the filtered manifest, then measured, then plain.

The filtered manifest removes long-answer rows while retaining measured
``grid_thw``. If it is absent, the measured manifest remains the preferred
fallback over the original plain manifest.

Run: python -m pytest tests/test_manifest_selection.py -q
"""
from vision_adapter.train import resolve_manifest_name


def test_prefers_the_filtered_manifest_when_present(tmp_path):
    (tmp_path / "train_manifest_filtered.jsonl").write_text("{}\n")
    (tmp_path / "train_manifest_grids.jsonl").write_text("{}\n")
    assert resolve_manifest_name(tmp_path) == "train_manifest_filtered.jsonl"


def test_falls_back_to_the_measured_manifest(tmp_path):
    (tmp_path / "train_manifest_grids.jsonl").write_text("{}\n")
    assert resolve_manifest_name(tmp_path) == "train_manifest_grids.jsonl"


def test_falls_back_to_the_plain_manifest(tmp_path):
    (tmp_path / "train_manifest.jsonl").write_text("{}\n")
    assert resolve_manifest_name(tmp_path) == "train_manifest.jsonl"


def test_no_manifest_at_all_is_an_error_not_a_default(tmp_path):
    """Silently defaulting would train on an empty plan."""
    import pytest

    with pytest.raises(FileNotFoundError, match="manifest"):
        resolve_manifest_name(tmp_path)


def test_filtered_manifest_precedes_measured_and_plain(tmp_path):
    (tmp_path / "train_manifest.jsonl").write_text("{}\n")
    (tmp_path / "train_manifest_grids.jsonl").write_text("{}\n")
    (tmp_path / "train_manifest_filtered.jsonl").write_text("{}\n")
    assert resolve_manifest_name(tmp_path) == "train_manifest_filtered.jsonl"


def test_stream_download_asks_for_the_measured_name(tmp_path):
    """A fresh run with no local file must fetch the same manifest the
    preference would have picked, or the two paths would disagree."""
    from vision_adapter.data.stream import MANIFEST_FILE

    assert MANIFEST_FILE == "train_manifest_grids.jsonl"
