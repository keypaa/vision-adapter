"""Does the loss actually depend on the visual span?

This is the question that decides whether any further training run is worth
GPU time. The historical run reached 111% of the Baseten grok reference
(64,000 samples, 0.55 epoch, loss ema 1.14) with a flat curve — that is a
long run producing no movement, and the candidates are all downstream of
this: if the loss barely moves when the image is removed, the projector is
not using the image and neither target_rms nor the sequence layout is the
first thing to fix.

Method: same batch, same projector, three variants —
  real   — the visual span as stored
  zero   — the visual span replaced with zeros
  shuffled — the visual span permuted across the batch
The gap between real and zero is the image's contribution to the loss.

Run: python -m pytest tests/test_visual_ablation.py -q
"""
import pytest
import torch


class _TinyModel(torch.nn.Module):
    """Stand-in backbone that actually reads its inputs_embeds.

    A real one is far away; what matters is that the harness can detect a
    difference when there is one, and the shapes/dtypes must match the
    production path (bf16 in, selective lm_head on -100-masked labels).
    """

    def __init__(self, vocab=32, dim=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.lm_head = torch.nn.Linear(dim, vocab)
        self.mix = torch.nn.Linear(dim, dim)

        class _Backbone(torch.nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def forward(self, inputs_embeds=None, attention_mask=None,
                        position_ids=None):
                # mix across the sequence so a changed visual span changes
                # the state that reaches the loss. position_ids is (4,B,L):
                # row 0 is (B,L) and must broadcast over the feature dim.
                h = inputs_embeds
                if position_ids is not None:
                    h = h + position_ids[0].unsqueeze(-1).to(h.dtype) * 0.01
                h = torch.tanh(self.outer.mix(h))
                return type("O", (), {"last_hidden_state": h})()

        self.model = _Backbone(self)

    def get_input_embeddings(self):
        return self.embed


def _batch(n_vis=4, dim=8, vocab=32, rows=1):
    return {
        "input_ids": torch.randint(0, vocab, (rows, n_vis + 6)),
        "attention_mask": torch.ones(rows, n_vis + 6, dtype=torch.long),
        "labels": torch.randint(0, vocab, (rows, n_vis + 6)),
        "vis": torch.randn(rows, n_vis, dim),
        "n_vis": torch.tensor([n_vis] * rows),
    }


def test_zeroing_the_visual_span_changes_the_loss():
    """If this fails, the harness cannot detect an image dependency at all."""
    from vision_adapter.train import _batch_loss

    torch.manual_seed(0)
    model = _TinyModel()
    proj = torch.nn.Linear(8, 8)
    batch = _batch()

    real = _batch_loss(model, proj, batch, "cpu", None)
    batch["vis"] = torch.zeros_like(batch["vis"])
    zeroed = _batch_loss(model, proj, batch, "cpu", None)
    assert real != zeroed, (
        "the loss is identical with and without the image — the harness is "
        "blind, or the projector genuinely ignores its input"
    )


def test_shuffling_the_visual_span_changes_the_loss():
    from vision_adapter.train import _batch_loss

    torch.manual_seed(1)
    model = _TinyModel()
    proj = torch.nn.Linear(8, 8)
    batch = _batch()

    real = _batch_loss(model, proj, batch, "cpu", None)
    batch["vis"] = batch["vis"].roll(1, dims=1)
    shuffled = _batch_loss(model, proj, batch, "cpu", None)
    assert real != shuffled


def test_the_ablation_reports_a_signed_image_contribution():
    """The helper must return the numbers, not just a boolean."""
    from vision_adapter.train import visual_ablation

    torch.manual_seed(2)
    model = _TinyModel()
    proj = torch.nn.Linear(8, 8)
    out = visual_ablation(model, proj, _batch(), "cpu")
    assert set(out) >= {"real", "zero", "shuffled"}
    assert out["real"] != out["zero"]
