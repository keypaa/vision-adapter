import json
from pathlib import Path

from vision_adapter.backends.local import LocalBackend
from vision_adapter.data.dataset import build_dataset


def test_dataset_writes_header_first(tmp_path: Path):
    b = LocalBackend(tmp_path)
    manifest = build_dataset(b, tmp_path, seed=0, limit=10, dry_run=True)
    assert manifest.exists()
    lines = manifest.read_text().splitlines()
    hdr = json.loads(lines[0])
    assert hdr["type"] == "manifest_header" and hdr["manifest_version"] == 1
    assert hdr["seeds"]["python"] == 0
    assert "agentic_source" in hdr["upstream"] or "upstream" in str(hdr)


def test_dataset_dry_run_deterministic(tmp_path: Path):
    b = LocalBackend(tmp_path)
    m1 = build_dataset(b, tmp_path / "run1", seed=42, limit=10, dry_run=True)
    m2 = build_dataset(b, tmp_path / "run2", seed=42, limit=10, dry_run=True)
    assert m1.read_bytes() == m2.read_bytes()
