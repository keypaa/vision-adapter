"""Held-out eval gate (Phase 3b grok check).

- forward_loss must equal the train-step loss on the same batch (same
  selective lm_head path, no backward/step).
- select_heldout_rows must only pick rows from never-trained shards,
  deterministically.
"""
import torch

from tests.test_adaptive_ckpt import _batch, _tiny_qwen
from vision_adapter.core import HourglassProjector


def test_forward_loss_matches_train_step():
    from vision_adapter.core import train_step_qwen
    from scripts.eval_heldout import forward_loss

    torch.manual_seed(0)
    model = _tiny_qwen()
    proj = HourglassProjector(4096, 32)
    batch = _batch([6, 6, 6, 6])
    loss_eval, tokens = forward_loss(model, proj, batch, "cpu")
    assert tokens > 0
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3)
    out = train_step_qwen(model, proj, opt, batch, "cpu", adaptive_ckpt=(10**9, 10**18))
    assert out["finite"]
    assert loss_eval == out["loss"]
    assert tokens == out["tokens"]


def test_forward_loss_deterministic():
    from scripts.eval_heldout import forward_loss

    torch.manual_seed(1)
    model = _tiny_qwen()
    proj = HourglassProjector(4096, 32)
    batch = _batch([5, 8, 3])
    assert forward_loss(model, proj, batch, "cpu") == forward_loss(model, proj, batch, "cpu")


def test_strip_trailing_eos_cuts_prefix_before_eos():
    from scripts.colab_unsloth_test import strip_trailing_eos

    # ids: [BOS?, img pad.., user.., EOS, pad..] — pad distinct from EOS here
    ids = torch.tensor([[1, 7, 7, 5, 6, 2, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    batch = {"input_ids": ids, "attention_mask": attn, "n_vis": torch.tensor([2])}
    gen_batch, cut = strip_trailing_eos(batch, eos_id=2)
    assert cut == 5
    assert gen_batch["input_ids"].shape == (1, 5)
    assert gen_batch["attention_mask"].shape == (1, 5)
    assert gen_batch["n_vis"].equal(batch["n_vis"])  # untouched keys pass through
    # Qwen case: pad_id == eos_id → first hit is still the real EOS (answer precedes pad)
    ids2 = torch.tensor([[1, 7, 5, 2, 2, 2]])
    batch2 = {"input_ids": ids2, "attention_mask": torch.ones(1, 6, dtype=torch.long)}
    _, cut2 = strip_trailing_eos(batch2, eos_id=2)
    assert cut2 == 3
    # no EOS at all → keep full length
    ids3 = torch.tensor([[1, 7, 5, 6]])
    batch3 = {"input_ids": ids3, "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    _, cut3 = strip_trailing_eos(batch3, eos_id=2)
    assert cut3 == 4


def test_select_heldout_rows_only_excluded_shards():
    from scripts.eval_heldout import select_heldout_rows

    rows = [
        {"emb": f"embeddings/k{i}.pt", "user": "u", "assistant": "a answer"}
        for i in range(10)
    ]
    index = {f"embeddings/k{i}.pt": ("data/emb_0000.parquet" if i % 2 else "data/emb_0005.parquet", i) for i in range(10)}
    held = {"data/emb_0000.parquet", "data/emb_0001.parquet"}
    sel = select_heldout_rows(rows, index, held, n=3, seed=0)
    assert len(sel) == 3
    assert all(index[r["emb"]][0] in held for r in sel)
    # deterministic across calls
    assert [r["emb"] for r in select_heldout_rows(rows, index, held, n=3, seed=0)] == [r["emb"] for r in sel]
    # capped at available
    assert len(select_heldout_rows(rows, index, held, n=100, seed=0)) == 5
