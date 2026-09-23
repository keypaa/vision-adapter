"""Run-scoped ckpt filenames (overwrite incident, Sept 2026).

The Sept-22 scaled run overwrote the Sept-13 hourglass step files (same
names, other architecture). Names are now optionally namespaced via
VISION_ADAPTER_CKPT_TAG; empty default reproduces legacy names exactly.
"""
import pytest


def test_default_tag_is_empty_and_legacy_names(monkeypatch):
    from vision_adapter.train import _ckpt_tag, _final_ckpt_filename, _step_ckpt_filename

    monkeypatch.delenv("VISION_ADAPTER_CKPT_TAG", raising=False)
    assert _ckpt_tag() == ""
    assert _step_ckpt_filename(400) == "projector_step400.pt"
    assert _final_ckpt_filename(4000) == "projector_final_4000.pt"


def test_tagged_names_and_bad_tag_rejected(monkeypatch):
    from vision_adapter.train import _ckpt_tag, _final_ckpt_filename, _step_ckpt_filename

    monkeypatch.setenv("VISION_ADAPTER_CKPT_TAG", "scaled")
    assert _ckpt_tag() == "scaled"
    assert _step_ckpt_filename(400) == "projector_scaled_step400.pt"
    assert _final_ckpt_filename(4000) == "projector_scaled_final_4000.pt"
    monkeypatch.setenv("VISION_ADAPTER_CKPT_TAG", "../evil")
    with pytest.raises(ValueError):
        _ckpt_tag()


def test_step_number_parses_both_forms():
    from vision_adapter.train import _ckpt_step_number

    assert _ckpt_step_number("projector_step400.pt") == 400
    assert _ckpt_step_number("projector_scaled_step400.pt") == 400
    assert _ckpt_step_number("projector_final_4000.pt") == -1
    assert _ckpt_step_number("probe_log.jsonl") == -1


def test_local_discovery_finds_tagged(tmp_path):
    from vision_adapter.train import _list_local_step_ckpts_desc

    (tmp_path / "projector_step100.pt").write_bytes(b"x")
    (tmp_path / "projector_scaled_step400.pt").write_bytes(b"x")
    (tmp_path / "projector_final_4000.pt").write_bytes(b"x")
    got = [p.name for p in _list_local_step_ckpts_desc(tmp_path)]
    assert got == ["projector_scaled_step400.pt", "projector_step100.pt"]


def test_find_local_ckpt_tagged_exact(tmp_path, monkeypatch):
    from vision_adapter.train import _find_local_ckpt

    monkeypatch.setenv("VISION_ADAPTER_CKPT_TAG", "scaled")
    (tmp_path / "projector_scaled_step400.pt").write_bytes(b"x")
    (tmp_path / "projector_step400.pt").write_bytes(b"y")
    assert _find_local_ckpt(tmp_path, 400).name == "projector_scaled_step400.pt"
