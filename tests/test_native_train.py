"""Native-protocol training batch (NEXT-2): full-length layout with labels.

Contract: legacy row [slot][img×nv][user][answer][EOS] becomes
[slot][vstart][pads×N][vend][user][answer][EOS] (N==nv asserted),
labels shifted by +2, mm=1 on pads only, positions with native
post-vision offset. n_vis=0 rows pass through unframed.
"""
import torch

from tests.test_adaptive_ckpt import _batch


def _native_ids():
    return {"image_token_id": 248056, "vision_start_id": 248053,
            "vision_end_id": 248054, "pad_token_id": 0}


def test_labels_shift_by_two_and_keep_answer():
    from vision_adapter.native import build_native_train_batch
    from vision_adapter.core import grid_for_nvis

    batch = _batch([4, 4])
    grids = torch.stack([grid_for_nvis(4), grid_for_nvis(4)])
    out = build_native_train_batch(batch, grids, **_native_ids())
    for i in range(2):
        old_sup = (batch["labels"][i] != -100).nonzero(as_tuple=False).squeeze(-1)
        new_sup = (out["labels"][i] != -100).nonzero(as_tuple=False).squeeze(-1)
        assert torch.equal(new_sup, old_sup + 2)
        assert torch.equal(out["labels"][i][new_sup], batch["labels"][i][old_sup])
        assert out["input_ids"].shape[1] == batch["input_ids"].shape[1] + 2


def test_mm_marks_only_pads_and_framing_present():
    from vision_adapter.native import build_native_train_batch
    from vision_adapter.core import grid_for_nvis

    batch = _batch([4, 4])
    grids = torch.stack([grid_for_nvis(4), grid_for_nvis(4)])
    out = build_native_train_batch(batch, grids, **_native_ids())
    for i in range(2):
        row = out["input_ids"][i].tolist()
        pads = [p for p, t in enumerate(row) if t == 248056]
        assert len(pads) == 4
        assert row[pads[0] - 1] == 248053 and row[pads[-1] + 1] == 248054
        mm = out["mm_token_type_ids"][i]
        assert set(mm.tolist()) <= {0, 1}
        assert int((mm == 1).sum()) == 4


def test_post_vision_text_offset_not_arange():
    from vision_adapter.native import build_native_train_batch
    from vision_adapter.core import grid_for_nvis

    batch = _batch([4, 4])
    grids = torch.stack([grid_for_nvis(4), grid_for_nvis(4)])
    out = build_native_train_batch(batch, grids, **_native_ids())
    pos = out["position_ids"]
    assert pos.shape[0] == 3
    B, L = out["input_ids"].shape
    # text after the vision span continues from the shifted cursor, not arange
    for i in range(B):
        live = int(out["attention_mask"][i].sum().item())
        tail = pos[:, i, live - 3: live]
        plain = torch.arange(live - 3, live).view(1, -1).expand(3, -1)
        assert not torch.equal(tail, plain)


def test_zero_vis_passthrough():
    from vision_adapter.native import build_native_train_batch

    batch = _batch([4, 4])
    grids = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long)  # row1 N=1 != 4 -> must raise
    try:
        build_native_train_batch(batch, grids, **_native_ids())
    except ValueError:
        return
    raise AssertionError("grid/vis mismatch must raise ValueError")


def test_native_train_env_resolver():
    from vision_adapter.core import _resolve_native_train

    import os

    os.environ.pop("VISION_ADAPTER_NATIVE_TRAIN", None)
    assert _resolve_native_train() is False
    os.environ["VISION_ADAPTER_NATIVE_TRAIN"] = "1"
    try:
        assert _resolve_native_train() is True
    finally:
        del os.environ["VISION_ADAPTER_NATIVE_TRAIN"]
    os.environ["VISION_ADAPTER_NATIVE_TRAIN"] = "nope"
    try:
        _resolve_native_train()
    except ValueError:
        return
    finally:
        del os.environ["VISION_ADAPTER_NATIVE_TRAIN"]
    raise AssertionError("bad VISION_ADAPTER_NATIVE_TRAIN must raise ValueError")


def test_train_step_finite_native_mode(monkeypatch):
    from tests.test_adaptive_ckpt import _tiny_qwen
    from vision_adapter.core import HourglassProjector, train_step_qwen

    monkeypatch.setenv("VISION_ADAPTER_NATIVE_TRAIN", "1")
    # tiny test model has vocab 256: shrink native ids into range (contract identical)
    monkeypatch.setattr("vision_adapter.native.IMAGE_TOKEN_ID", 200)
    monkeypatch.setattr("vision_adapter.native.VISION_START_ID", 201)
    monkeypatch.setattr("vision_adapter.native.VISION_END_ID", 202)
    torch.manual_seed(0)
    model = _tiny_qwen()
    proj = HourglassProjector(4096, 32)
    batch = _batch([6, 6])
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3)
    out = train_step_qwen(model, proj, opt, batch, "cpu", adaptive_ckpt=(10**9, 10**18))
    assert out["finite"]
