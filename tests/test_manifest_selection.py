"""train picks the measured-geometry manifest when it exists, else the old one.

The backfilled manifest (grid_thw on every row, verified 117600/117600 against
the stored n_vis) is uploaded alongside the original rather than over it: the
old one is what existing checkpoints recorded a manifest_sha256 against, and
a resume that silently saw a different manifest would be unverifiable.

So the two coexist and the runner prefers the measured one. No flag, no
config — a run picks up the measured geometry as soon as the file is there,
and grid_source in the run card says which one it actually read.

Run: python -m pytest tests/test_manifest_selection.py -q
"""
from vision_adapter.train import resolve_manifest_name


def test_prefers_the_backfilled_manifest_when_present(tmp_path):
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


def test_preference_is_stable_regardless_of_creation_order(tmp_path):
    """Both present: the measured one wins even if it is older."""
    (tmp_path / "train_manifest_grids.jsonl").write_text("{}\n")
    (tmp_path / "train_manifest.jsonl").write_text("{}\n")
    assert resolve_manifest_name(tmp_path) == "train_manifest_grids.jsonl"


def test_stream_download_asks_for_the_measured_name(tmp_path):
    """A fresh run with no local file must fetch the same manifest the
    preference would have picked, or the two paths would disagree."""
    from vision_adapter.data.stream import MANIFEST_FILE

    assert MANIFEST_FILE == "train_manifest_grids.jsonl"
