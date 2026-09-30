"""Generation must position the visual span the way training did.

Audit finding (2026-09-30): `train_step_qwen` builds mRoPE over the visual
span `[1:1+n_vis]` and feeds it as `position_ids`. The default generation
branch in `scripts/colab_unsloth_test.py` called `model.generate` with
`inputs_embeds` and `attention_mask` only — no `position_ids` at all. So the
image tokens sat in the same slots carrying plain `arange` positions, i.e. a
geometry the backbone was never trained to interpret.

That is a direct train/inference mismatch and a sufficient cause of "trained
fine, generates nothing usable". The held-out loss gate is faithful (it calls
`train_position_ids`), which is exactly why the loss curve and the generation
check disagreed with each other.

Run: python -m pytest tests/test_gen_positions.py -q
"""
import torch

from vision_adapter.core import _resolve_positions_mode, train_position_ids


class _Tok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(ord(c) % 900) + 10 for c in text] or [10]}


def _batch(n_vis=8):
    from vision_adapter.core import make_collate

    coll = make_collate(_Tok(), 0, max_len=64, vision_dim=8)
    return coll([{"vis": torch.randn(n_vis, 8), "user": "click here",
                  "assistant": "", "g": "test"}])


def test_legacy_generation_must_receive_mrope_positions():
    """The batch a generator builds must carry the same positions training used."""
    batch = _batch()
    assert _resolve_positions_mode() == "mrope", "precondition: training uses mRoPE"

    pos = train_position_ids(batch)
    # the visual span positions are not plain arange — that is the whole point
    span = pos[1, 0, 1:5]
    assert not torch.equal(span, torch.arange(1, 5)), (
        "positions collapsed to arange: generation would see a geometry the "
        "backbone never trained on"
    )


def test_generation_kwargs_carry_position_ids():
    """model.generate must be handed position_ids, not just embeds+mask."""
    from scripts.colab_unsloth_test import build_legacy_gen_kwargs

    batch = _batch()
    out = build_legacy_gen_kwargs(batch, torch.device("cpu"))
    assert "position_ids" in out, (
        "legacy generation builds kwargs without position_ids — the defect"
    )
    pos = out["position_ids"]
    assert pos.shape[0] == 4, "mRoPE is 4-row (arange + t/h/w)"


def test_the_legacy_branch_uses_the_position_helper():
    """If the default branch changes, revisit this — do not just delete it."""
    import inspect

    from scripts.colab_unsloth_test import build_legacy_gen_kwargs

    # the legacy branch must not call generate() without the positions helper
    src = inspect.getsource(__import__("scripts.colab_unsloth_test", fromlist=["x"]))
    assert "build_legacy_gen_kwargs(" in src, (
        "the legacy generation branch stopped using build_legacy_gen_kwargs — "
        "if that was deliberate, revisit this pin"
    )
    assert callable(build_legacy_gen_kwargs)
