"""ScaledHourglassProjector (U4 mechanism fix, NEXT-5).

Measured on Molab: raw projector outputs rms=16.42 vs Qwen3.5-2B table
rms=0.0131 (~1250x) -> saturated attention, dead generation. This variant
appends an output LayerNorm x fixed target RMS. Train path untouched
(selected via env, default hourglass).
"""
import copy

import pytest
import torch

from tests.test_adaptive_ckpt import _batch, _tiny_qwen
from vision_adapter.core import HourglassProjector, build_projector


def test_builder_defaults_to_hourglass(monkeypatch):
    monkeypatch.delenv("VISION_ADAPTER_PROJECTOR", raising=False)
    assert type(build_projector(4096, 32)).__name__ == "HourglassProjector"


def test_builder_scaled_variant_and_bad_value(monkeypatch):
    from vision_adapter.core import ScaledHourglassProjector

    monkeypatch.setenv("VISION_ADAPTER_PROJECTOR", "scaled")
    proj = build_projector(4096, 32)
    assert isinstance(proj, ScaledHourglassProjector)
    assert isinstance(proj.base, HourglassProjector)
    monkeypatch.setenv("VISION_ADAPTER_PROJECTOR", "nope")
    with pytest.raises(ValueError):
        build_projector(4096, 32)


def test_output_rms_matches_target():
    from vision_adapter.core import ScaledHourglassProjector

    torch.manual_seed(0)
    proj = ScaledHourglassProjector(4096, 32, target_rms=0.02).to(torch.float32)
    x = torch.randn(4, 12, 4096)
    with torch.no_grad():
        y = proj(x)
    assert y.shape == (4, 12, 32)
    assert float(y.pow(2).mean().sqrt()) == pytest.approx(0.02, rel=0.05)
    assert float(y.abs().max()) < 1.0  # vs 172 measured on the raw head


def test_grads_flow_and_deterministic():
    from vision_adapter.core import ScaledHourglassProjector

    torch.manual_seed(1)
    proj = ScaledHourglassProjector(4096, 32).to(torch.float32)
    x = torch.randn(2, 5, 4096)
    y1 = proj(x)
    y1.sum().backward()
    assert all(p.grad is not None for p in proj.parameters())
    for p in proj.parameters():
        p.grad = None
    torch.manual_seed(1)
    proj2 = ScaledHourglassProjector(4096, 32).to(torch.float32)
    proj2.load_state_dict(copy.deepcopy(proj.state_dict()))
    assert torch.equal(proj(x), proj2(x))


def test_train_step_finite_with_scaled_head():
    from vision_adapter.core import ScaledHourglassProjector, train_step_qwen

    torch.manual_seed(2)
    model = _tiny_qwen()
    proj = ScaledHourglassProjector(4096, 32).to(torch.float32)
    batch = _batch([6, 6, 6, 6])
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3)
    out = train_step_qwen(model, proj, opt, batch, "cpu", adaptive_ckpt=(10**9, 10**18))
    assert out["finite"]


def test_streaming_train_uses_builder():
    import pathlib

    src = pathlib.Path("vision_adapter/train.py").read_text()
    assert "build_projector" in src
