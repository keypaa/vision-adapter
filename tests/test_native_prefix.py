"""Native-protocol prefix builder (U1 expansion rule) — CPU-only TDD tests.

Rule under test (all verified against transformers 5.12.1 source, local
site-packages — never from memory):
- placeholders/image N = prod(grid_thw) // merge_size**2
  (processing_qwen3_vl.Qwen3VLProcessor.replace_image_token +
  modeling_qwen3_vl.Qwen3VLModel.get_image_features split_sizes;
  merge_size default 2 = vision_config.spatial_merge_size)
- framing: <vision_start> immediately before first <image_pad>,
  <vision_end> immediately after last one
  (modeling_qwen3_vl._get_image_nums_and_video_nums counts an image only
  where vision_start directly precedes an image pad; video path builds
  vision_start + pads + vision_end per frame)
- mm_token_type_ids: 0 text / 1 image / 2 video
  (processing_utils.create_mm_token_type_ids; processor image_token_ids
  contains ONLY image_token_id, so vision_start/end stay 0)
- get_placeholder_mask: #image_token_id in input_ids must equal the
  scattered image_embeds length, else error
- compute_3d_position_ids: raises when multimodal grids are passed
  without mm_token_type_ids; vision spans consume next grid and advance
  current_pos by max(h, w) // spatial_merge_size

Hermetic: stub tokenizer + collated batch fixtures from
tests/test_adaptive_ckpt (same import pattern as test_eval_heldout),
tiny tensors, CPU only. No GPU, no training-path changes.
"""
import torch

from tests.test_adaptive_ckpt import StubTok, _batch, _tiny_qwen  # noqa: F401  (fixture reuse)
from scripts.native_prefix import build_native_prefix, placeholder_count


IMAGE_ID = 248056
VSTART_ID = 248053
VEND_ID = 248054


class NativeTok(StubTok):
    bos_token_id = None  # Qwen-like: no BOS (collate leaves slot 0 as pad)
    image_token_id = IMAGE_ID
    vision_start_token_id = VSTART_ID
    vision_end_token_id = VEND_ID


def _native_batch(n_vis_list, tok=None):
    tok = tok or NativeTok()
    return _batch(n_vis_list, tok=tok), tok


def _grids_for(n_vis_list):
    # hand-built (t, h, w) grids with prod // 4 == n_vis (merge_size=2)
    table = {4: (1, 4, 4), 8: (1, 4, 8), 5: (1, 2, 10), 3: (1, 2, 6)}
    return torch.tensor([table[n] for n in n_vis_list], dtype=torch.long)


def test_placeholder_count_matches_rule():
    assert placeholder_count(torch.tensor([1, 4, 4]), 2) == 4
    assert placeholder_count(torch.tensor([1, 4, 8]), 2) == 8
    assert placeholder_count(torch.tensor([2, 4, 4]), 2) == 8  # t=2 frames
    assert placeholder_count(torch.tensor([1, 4, 4]), 1) == 16  # merge=1: no downscale


def test_placeholder_count_in_prefix_equals_rule():
    batch, tok = _native_batch([4, 8])
    out = build_native_prefix(batch, tok, _grids_for([4, 8]))
    for i, n in enumerate([4, 8]):
        row = out["input_ids"][i].tolist()
        assert row.count(IMAGE_ID) == n == placeholder_count(_grids_for([4, 8])[i], 2)


def test_mm_token_type_ids_mark_only_image_pads():
    batch, tok = _native_batch([4, 8])
    out = build_native_prefix(batch, tok, _grids_for([4, 8]))
    mm = out["mm_token_type_ids"]
    assert mm.shape == out["input_ids"].shape
    for i in range(2):
        row = out["input_ids"][i].tolist()
        mrow = mm[i].tolist()
        for pos, (t, m) in enumerate(zip(row, mrow)):
            if t == IMAGE_ID:
                assert m == 1, f"image pad at {pos} must be type 1"
            else:
                assert m == 0, f"token {t} at {pos} must be type 0 (text)"
        # vision_start/end framing tokens are text-typed, never image-typed
        vs_pos = row.index(VSTART_ID)
        ve_pos = row.index(VEND_ID)
        assert mrow[vs_pos] == 0 and mrow[ve_pos] == 0


def test_determinism():
    batch, tok = _native_batch([5, 8, 3])
    grids = _grids_for([5, 8, 3])
    out1 = build_native_prefix(batch, tok, grids)
    out2 = build_native_prefix(batch, tok, grids)
    for k in ("input_ids", "attention_mask", "mm_token_type_ids", "position_ids"):
        assert torch.equal(out1[k], out2[k]), f"{k} must be deterministic"


def test_vision_start_end_framing():
    batch, tok = _native_batch([4, 8])
    out = build_native_prefix(batch, tok, _grids_for([4, 8]))
    for i in range(2):
        row = out["input_ids"][i].tolist()
        pads = [p for p, t in enumerate(row) if t == IMAGE_ID]
        assert len(pads) >= 1
        assert pads == list(range(pads[0], pads[-1] + 1)), "pads must be contiguous"
        assert row[pads[0] - 1] == VSTART_ID, "vision_start directly before first pad"
        assert row[pads[-1] + 1] == VEND_ID, "vision_end directly after last pad"


def test_no_eos_truncation_safety():
    batch, tok = _native_batch([4, 8])
    grids = _grids_for([4, 8])
    out = build_native_prefix(batch, tok, grids)
    eos = tok.eos_token_id
    for i in range(2):
        mask = out["attention_mask"][i].bool()
        live = out["input_ids"][i][mask].tolist()
        assert len(live) > 0
        assert live[-1] != eos, "generation prefix must not end on EOS"
        assert live.count(eos) == 0, "answer EOS must be stripped from the prefix"
    # answer stripped => prefix strictly shorter than the supervised sequence
    assert out["input_ids"].shape[1] < batch["input_ids"].shape[1]


def test_position_ids_are_3d_mrope():
    batch, tok = _native_batch([4, 8])
    out = build_native_prefix(batch, tok, _grids_for([4, 8]))
    pos = out["position_ids"]
    assert pos.shape == (3, 2, out["input_ids"].shape[1])
    # vision span must carry 3D (t/h/w) positions, not a plain arange:
    # the three rows differ somewhere inside the image span
    for i in range(2):
        row = out["input_ids"][i].tolist()
        pads = [p for p, t in enumerate(row) if t == IMAGE_ID]
        span = pos[:, i, pads[0]: pads[-1] + 1]
        assert not torch.equal(span[1], span[2]), "h/w rows must differ over vision span"


def test_grid_vis_mismatch_raises():
    batch, tok = _native_batch([4, 8])
    bad = torch.tensor([[1, 4, 4], [1, 2, 6]], dtype=torch.long)  # row1 => 3 != 8
    try:
        build_native_prefix(batch, tok, bad)
    except ValueError:
        return
    raise AssertionError("grid/vis length mismatch must raise ValueError")


def test_training_path_untouched():
    import pathlib

    core_src = pathlib.Path("vision_adapter/core.py").read_text()
    train_src = pathlib.Path("vision_adapter/train.py").read_text()
    for src, name in ((core_src, "core.py"), (train_src, "train.py")):
        assert "248056" not in src, f"{name} must not contain native placeholder ids"
        assert "mm_token_type_ids" not in src, f"{name} must not contain native mm ids"
        assert "native_prefix" not in src, f"{name} must not import the eval-only helper"
