"""Fetched manifest must land on disk so later stages never FileNotFoundError."""

from vision_adapter.train import _persist_fetched_manifest
from vision_adapter.manifest import load_manifest


def test_persist_fetched_manifest_roundtrip(tmp_path):
    rows = [
        {"emb": "e1", "user": "u1", "assistant": "a1", "g": "t"},
        {"emb": "e2", "user": "u2", "assistant": "a2", "g": "t"},
    ]
    out = _persist_fetched_manifest(tmp_path, rows)
    assert out == tmp_path / "train_manifest.jsonl"
    back, header = load_manifest(out)
    assert len(back) == 2
    assert header is not None  # header-first, not legacy
