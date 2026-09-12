"""Adaptive per-batch grad checkpointing (OOM fix for worst-bucket batches).

Pins the fresh-eyes review holes:
- Hole 3: trigger must gate on padded shape (B·L²), NOT on attention_mask sum.
  A batch with one L=4000 row padded out must flip ON even when another
  batch has the same mask sum spread over short rows.
- Semantics: ckpt ON vs OFF must give identical loss + projector grads.
- Hygiene: toggle must restore the flag (even on exception) and leak no hooks.

Hermetic: stub tokenizer + tiny random-weight Qwen3.5 on CPU, no GPU.
"""
import copy

import pytest
import torch

from vision_adapter.core import (
    HourglassProjector,
    ckpt_needed_for_batch,
    make_collate,
    train_step_qwen,
)


class StubTok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        ids = [(ord(c) % 900) + 10 for c in text]
        return {"input_ids": ids or [10]}


def _tiny_qwen(layers: int = 2, vocab: int = 256, hidden: int = 32):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
    tc = cfg.get_text_config()
    tc.vocab_size = vocab
    tc.hidden_size = hidden
    tc.intermediate_size = 2 * hidden
    tc.num_hidden_layers = layers
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.layer_types = ["linear_attention"] * (layers - 1) + ["full_attention"]
    try:
        tc.mtp_num_hidden_layers = None
    except Exception:
        pass
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
    model.config.use_cache = False
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _batch(n_vis_list, tok=None):
    tok = tok or StubTok()
    items = [
        {"vis": torch.randn(nv, 4096), "user": f"u{i}", "assistant": f"a{i} answer", "g": "t"}
        for i, nv in enumerate(n_vis_list)
    ]
    return make_collate(tok, tok.pad_token_id, max_len=8192)(items)


# --- trigger: shape-gated, not sum-gated (Hole 3) ---


def test_trigger_off_for_small_batch():
    b = _batch([50, 60, 40])
    assert ckpt_needed_for_batch(b, l_max=2500) is False


def test_trigger_on_for_long_batch():
    b = _batch([10, 5000, 12])
    assert ckpt_needed_for_batch(b, l_max=2500) is True


def test_trigger_uses_shape_not_mask_sum():
    # Same mask sum (~8000 real tokens), different shapes:
    # one L=4002 batch must flip ON, one L=1002 batch stays OFF.
    tall = _batch([4000, 4000])
    wide = _batch([1000] * 8)
    assert tall["attention_mask"].sum() == pytest.approx(wide["attention_mask"].sum(), rel=0.05)
    assert ckpt_needed_for_batch(tall, l_max=2500) is True
    assert ckpt_needed_for_batch(wide, l_max=2500) is False


def test_trigger_cost_gate():
    # L under l_max but B·L² over cost_max still flips ON.
    b = _batch([1000] * 16)
    B, L = b["input_ids"].shape
    cost = B * L * L
    assert L < 2500  # L-gate alone would stay OFF
    assert ckpt_needed_for_batch(b, l_max=2500, cost_max=cost - 1) is True
    assert ckpt_needed_for_batch(b, l_max=2500, cost_max=cost + 1) is False


# --- semantics: ON == OFF (grads + loss identical) ---


def _run_step(model, proj, batch):
    for p in proj.parameters():
        p.grad = None
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3)
    return train_step_qwen(model, proj, opt, batch, "cpu")


def test_ckpt_on_off_grads_identical():
    torch.manual_seed(0)
    model = _tiny_qwen()
    proj = HourglassProjector(4096, 32)
    batch = _batch([5, 8, 3])
    state = copy.deepcopy(proj.state_dict())

    proj.load_state_dict(state)
    out_off = train_step_qwen(
        model, proj, torch.optim.AdamW(proj.parameters(), lr=1e-3),
        batch, "cpu", adaptive_ckpt=(10**9, 10**18),  # force OFF
    )
    grads_off = [p.grad.detach().clone() for p in proj.parameters()]

    proj.load_state_dict(state)
    out_on = train_step_qwen(
        model, proj, torch.optim.AdamW(proj.parameters(), lr=1e-3),
        batch, "cpu", adaptive_ckpt=(1, 1),  # force ON
    )
    grads_on = [p.grad.detach().clone() for p in proj.parameters()]

    assert out_on["loss"] == pytest.approx(out_off["loss"], abs=1e-6)
    for g_on, g_off in zip(grads_on, grads_off):
        assert torch.allclose(g_on, g_off, atol=1e-6)


def test_toggle_restores_flag_and_leaks_no_hooks():
    torch.manual_seed(1)
    model = _tiny_qwen()
    proj = HourglassProjector(4096, 32)
    batch = _batch([5, 8, 3])
    n_hooks_before = len(model.get_input_embeddings()._forward_hooks)
    assert model.is_gradient_checkpointing is False

    for _ in range(5):
        _run_step(model, proj, batch)

    assert model.is_gradient_checkpointing is False
    assert len(model.get_input_embeddings()._forward_hooks) == n_hooks_before


def test_step_reports_ckpt_decision():
    torch.manual_seed(2)
    model = _tiny_qwen()
    proj = HourglassProjector(4096, 32)
    small = _batch([5, 8, 3])
    big = _batch([10, 3000, 12])
    out_small = _run_step(model, proj, small)
    out_big = _run_step(model, proj, big)
    assert out_small["ckpt_on"] is False
    assert out_big["ckpt_on"] is True


def test_none_disables_adaptive_ckpt_entirely():
    torch.manual_seed(3)
    model = _tiny_qwen()
    proj = HourglassProjector(4096, 32)
    big = _batch([10, 3000, 12])
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3)
    out = train_step_qwen(model, proj, opt, big, "cpu", adaptive_ckpt=None)
    assert out["finite"]
    assert out["ckpt_on"] is False
    assert model.is_gradient_checkpointing is False


def test_env_overrides_thresholds(monkeypatch):
    import vision_adapter.core as core

    monkeypatch.setenv("VISION_ADAPTER_L_MAX", "100")
    monkeypatch.setenv("VISION_ADAPTER_COST_MAX", "10")
    assert core._resolve_ckpt_budget("auto") == (100, 10)
    monkeypatch.delenv("VISION_ADAPTER_L_MAX")
    monkeypatch.delenv("VISION_ADAPTER_COST_MAX")
    assert core._resolve_ckpt_budget("auto") == (
        core.DEFAULT_L_MAX,
        core.DEFAULT_COST_MAX,
    )
    assert core._resolve_ckpt_budget(None) is None
    assert core._resolve_ckpt_budget((7, 9)) == (7, 9)


def test_bl2_separates_killer_from_safe_at_equal_mask_sum():
    """Offline replay (#5): at ~equal mask sums, B·L² must separate the
    killer shape (one huge n_vis + padding) from the safe shape by ~10×+.

    Regression pin: if collate ever stops padding to max(n_vis), revisit.
    """
    safe = _batch([700] * 16)
    killer = _batch([5000] + [440] * 15)
    sum_safe = int(safe["attention_mask"].sum())
    sum_killer = int(killer["attention_mask"].sum())
    assert abs(sum_killer - sum_safe) / sum_safe < 0.05
    bs, ls = safe["input_ids"].shape
    bk, lk = killer["input_ids"].shape
    ratio_sum = sum_killer / sum_safe
    ratio_bl2 = (bk * lk * lk) / (bs * ls * ls)
    assert ratio_bl2 > 10 * ratio_sum


def test_save_due_step_gate_and_time_gate():
    from vision_adapter.train import _save_due

    # step gate: fires on multiples, regardless of time
    assert _save_due(100, 100, 0.0, 60.0) is True
    assert _save_due(50, 100, 0.0, 60.0) is False
    # time gate: 10min without save forces a save mid-block (monster steps)
    assert _save_due(50, 100, 0.0, 600.0) is True
    assert _save_due(50, 100, 0.0, 599.9) is False
    # custom interval
    assert _save_due(50, 100, 0.0, 300.0, max_interval_s=300) is True


def test_expandable_segments_defaulted_but_never_overridden(monkeypatch):
    import os

    from vision_adapter.train import _ensure_expandable_segments

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert _ensure_expandable_segments() is True
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    assert _ensure_expandable_segments() is False
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"
