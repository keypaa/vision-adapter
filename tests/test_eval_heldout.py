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


def test_gen_kwargs_card_matches_model_card_vl_recipe():
    from scripts.colab_unsloth_test import build_gen_kwargs

    greedy = build_gen_kwargs("greedy")
    assert greedy == {"do_sample": False}
    card = build_gen_kwargs("card")
    # Qwen3.5 model-card recipe, non-thinking VL tasks
    assert card == {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


def test_native_mode_resolves_conditional_generation_class():
    import inspect

    from scripts.colab_unsloth_test import resolve_qwen_class

    causal = resolve_qwen_class(native=False)
    native = resolve_qwen_class(native=True)
    assert causal.__name__ == "AutoModelForCausalLM"
    assert native.__name__ == "Qwen3_5ForConditionalGeneration"
    params = inspect.signature(native.forward).parameters
    for kw in ("inputs_embeds", "mm_token_type_ids", "image_grid_thw", "position_ids"):
        assert kw in params, f"conditional forward must accept {kw}"


def test_manual_generate_mechanics_cpu():
    from scripts.colab_unsloth_test import manual_generate

    torch.manual_seed(0)
    model = _tiny_qwen()
    model.eval()
    B, L, H = 1, 12, 32
    embeds = torch.randn(B, L, H)
    mask = torch.ones(B, L, dtype=torch.long)
    out1 = manual_generate(model, embeds, mask, max_new_tokens=5, mode="greedy", eos_id=2, seed=0)
    out2 = manual_generate(model, embeds, mask, max_new_tokens=5, mode="greedy", eos_id=0 - 1, seed=0)
    assert len(out1) <= 5 and len(out2) == 5  # greedy stops only on real EOS
    card1 = manual_generate(model, embeds, mask, max_new_tokens=5, mode="card", eos_id=-1, seed=0)
    card2 = manual_generate(model, embeds, mask, max_new_tokens=5, mode="card", eos_id=-1, seed=0)
    assert card1 == card2  # seeded sampling is deterministic
    assert len(card1) == 5


def test_score_text_nll_structure_and_sensitivity():
    from scripts.eval_heldout import score_text_nll

    torch.manual_seed(0)
    model = _tiny_qwen()
    proj = HourglassProjector(4096, 32)
    vis_a = torch.randn(6, 4096)
    vis_b = torch.randn(6, 4096)
    user, assistant = "u test", "a answer here"
    tok = _stub_tok()
    nll_same = score_text_nll(model, proj, vis_a, user, assistant, tok, "cpu")
    nll_same2 = score_text_nll(model, proj, vis_a, user, assistant, tok, "cpu")
    nll_swap = score_text_nll(model, proj, vis_b, user, assistant, tok, "cpu")
    assert nll_same == nll_same2  # deterministic
    assert nll_same > 0 and nll_swap > 0  # finite positive NLLs
    assert nll_same != nll_swap  # vis actually conditions the score


def test_discrimination_pairs_cover_rows():
    from scripts.eval_heldout import discrimination_pairs

    rows = [{"emb": f"e{i}"} for i in range(5)]
    pairs = discrimination_pairs(rows)
    assert pairs == [(rows[0], rows[1]), (rows[2], rows[3])]  # leftover dropped


def _stub_tok():
    from tests.test_adaptive_ckpt import StubTok

    return StubTok()


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
