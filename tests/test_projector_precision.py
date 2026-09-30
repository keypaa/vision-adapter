"""The trainable projector must be fp32 whatever the backbone's dtype.

Measured on the real 200-step checkpoint series (audit 2026-09-30): the
projector's input LayerNorm gain, `ln.weight`, was still exactly 1.0 after
200 steps while the other five tensors moved (up.weight delta 12.88, ln.bias
0.64). Cause: the projector was built in bf16, where the spacing at magnitude
1.0 is 0.0078 (down-step 0.0039) and the AdamW update is lr=5e-4 — the update
rounds to zero on every step. Isolated reproduction: 500 steps with large
grads, 0 of 4096 elements changed.

Only the fp16 branch ever got an fp32 projector, so the fix left the common
case (bf16) broken.

The projector is the only trainable module, so it is the one place where
precision actually matters: the backbone can stay bf16, the thing being
optimised should not.

Run: python -m pytest tests/test_projector_precision.py -q
"""
import torch

from vision_adapter.train import resolve_proj_dtype


def test_projector_is_fp32_on_bf16_backbone():
    assert resolve_proj_dtype("cuda", torch.bfloat16) == torch.float32


def test_projector_is_fp32_on_fp16_backbone():
    assert resolve_proj_dtype("cuda", torch.float16) == torch.float32


def test_projector_is_fp32_on_fp32_backbone():
    assert resolve_proj_dtype("cuda", torch.float32) == torch.float32


def test_projector_is_fp32_on_cpu():
    assert resolve_proj_dtype("cpu", torch.float32) == torch.float32


def test_a_5e4_lr_step_is_representable_in_the_projector_dtype():
    """The reason, stated as a test: bf16 cannot represent the update.

    Only bf16 among the three is coarser than the step — fp16 does move at
    1e-3 spacing, which is why the fp16 path was never the visible symptom.
    """
    lr = 5e-4
    bf16 = torch.ones(4, dtype=torch.bfloat16)
    assert torch.equal((bf16 - lr).to(torch.bfloat16), bf16), (
        "bf16 rounds a 5e-4 update away — the tensor never moves"
    )
    for dt in (torch.float16, torch.float32):
        param = torch.ones(4, dtype=dt)
        assert not torch.equal((param - lr).to(dt), param), dt
