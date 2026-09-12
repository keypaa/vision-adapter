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


def test_streaming_failure_returns_1_without_smoke(tmp_path, monkeypatch):
    import vision_adapter.train as tr
    from vision_adapter.config import probe_config

    monkeypatch.setattr(tr, "_streaming_train", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    def _no_smoke(*a, **k):
        raise AssertionError("smoke fallback must not run after a real streaming failure")
    monkeypatch.setattr(tr, "_smoke_train_with_fake_data", _no_smoke)

    rc = tr.run_train(tmp_path, probe_config(), backend=None, max_steps=5, device="cpu")
    assert rc == 1
