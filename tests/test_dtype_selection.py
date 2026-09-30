"""Backbone dtype selection: fp16 NaNs the projector grads, bf16 does not.

Measured on a T4 (cc 7.5) 2026-09-30, 364-token visual span, projector fp32:
    fp16: loss=9.36 gnorm=nan  nonfinite_params=6/6
    bf16: loss=9.21 gnorm=522   nonfinite_params=0/6
The forward is fine either way (hidden absmax 48, logits 10.7) — it is the
backward through the frozen backbone that overflows, because fp16 grads
saturate at 65504 while the projector is the only fp32 module. A GradScaler
does not help: the overflowing values are not parameter grads of an fp16
module, they are activations flowing back to the fp32 head.

So the fix is to stop auto-selecting fp16. Only pre-Ampere (cc<70) falls back
to fp32, where bf16 is not available at all.

Run: python -m pytest tests/test_dtype_selection.py -q
"""
import pytest
import torch

from vision_adapter.train import resolve_dtype


@pytest.mark.parametrize(
    "cc,expected",
    [
        (75, torch.bfloat16),    # T4 — fp16 NaN'd the projector grads
        (70, torch.bfloat16),    # V100
        (80, torch.bfloat16),    # A100
        (86, torch.bfloat16),    # RTX 30xx
        (120, torch.bfloat16),   # Blackwell
        (60, torch.float32),     # P100 / Pascal: no bf16
        (61, torch.float32),
    ],
)
def test_auto_dtype_never_picks_fp16(cc, expected):
    assert resolve_dtype(cc, "auto") == expected


def test_explicit_fp16_is_still_honoured():
    """The escape hatch stays; it just isn't the default."""
    assert resolve_dtype(75, "fp16") == torch.float16
    assert resolve_dtype(75, "bf16") == torch.bfloat16
    assert resolve_dtype(75, "fp32") == torch.float32


def test_unknown_dtype_arg_rejects_loudly():
    with pytest.raises(ValueError, match="dtype"):
        resolve_dtype(75, "int8")


def test_both_training_paths_share_the_one_selection():
    """Regression: the local and streaming paths had diverged (fp32 vs fp16
    on a T4), which is how the NaN hid in only one of them."""
    src = __import__("pathlib").Path(
        __import__("vision_adapter.train", fromlist=["__file__"]).__file__
    ).read_text()
    # exactly one dtype_map literal should remain, in the shared helper
    assert src.count('"fp16": _torch.float16') + src.count('"fp16": torch.float16') == 1
