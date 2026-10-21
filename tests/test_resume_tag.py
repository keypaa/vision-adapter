"""A resume must pick THIS run's checkpoint, not the previous run's.

The HF ckpt repo and the local cache both hold the old probe's untagged
files (projector_step4000.pt). With a run tag set, "the latest checkpoint"
has to mean the latest of THIS run -- otherwise a resume silently loads the
old run's final checkpoint (which the recipe guard then refuses, so the run
dies for a confusing reason instead of resuming).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_adapter.train import _find_local_ckpt


def _mk(d: Path, name: str):
    p = d / name
    p.write_bytes(b"x")
    return p


def test_the_local_resume_prefers_the_tagged_run(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_ADAPTER_CKPT_TAG", "p05")
    _mk(tmp_path, "projector_step4000.pt")      # the OLD run's final
    _mk(tmp_path, "projector_p05_step2300.pt")  # THIS run's latest
    got = _find_local_ckpt(tmp_path)
    assert got.name == "projector_p05_step2300.pt"


def test_without_a_tag_the_latest_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("VISION_ADAPTER_CKPT_TAG", raising=False)
    _mk(tmp_path, "projector_step4000.pt")
    _mk(tmp_path, "projector_step2300.pt")
    got = _find_local_ckpt(tmp_path)
    assert got.name == "projector_step4000.pt"


def test_an_exact_step_ignores_the_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_ADAPTER_CKPT_TAG", "p05")
    _mk(tmp_path, "projector_step4000.pt")
    _mk(tmp_path, "projector_p05_step2300.pt")
    got = _find_local_ckpt(tmp_path, step=2300)
    assert got.name == "projector_p05_step2300.pt"
