"""CPU smoke test for grok_probe_qwen.py: the whole recipe end-to-end with a
tiny random-weight Qwen3.5 (2 layers, fp32) and fake cached embeddings.

Hermetic — no HF downloads, no GPU. Pins:
  - shapes flow end-to-end ([B,maxv,4096] vis -> projector -> inputs_embeds)
  - loss is finite and VARIES across steps (projector actually receives grads)
  - collate layout/masking matches test_train_collate.py's pins
  - resume artifacts round-trip (safetensors + opt state)

Run: python -m pytest test_grok_probe_smoke.py -q
"""
import pytest
import torch

import grok_probe_qwen as gp


# --------------------------- hermetic fixtures -------------------------------


class StubTok:
    """Char-level ids >= 10 so they never collide with pad=0/bos=1/eos=2."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        ids = [(ord(c) % 900) + 10 for c in text]
        return {"input_ids": ids or [10]}


def _tiny_qwen(layers: int = 4, vocab: int = 1024, hidden: int = 64):
    """Random-weight Qwen3.5 text backbone, fp32 CPU — structure identical to
    Qwen/Qwen3.5-2B's text stack (embed -> decoder layers -> lm_head).

    The last layer is full_attention ON PURPOSE (mirrors the real model's
    every-4th-layer pattern): a stack of ONLY linear_attention layers gates
    context away so completely that the supervised loss stops responding to
    earlier positions (~5e-7 change on a +100 perturbation) and the projector
    would appear dead. Verified empirically; see test_grads_reach_projector."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
    tc = cfg.get_text_config()
    tc.vocab_size = vocab
    tc.hidden_size = hidden
    tc.intermediate_size = 2 * hidden
    tc.num_hidden_layers = layers
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    # linear-attention layer sizes scale off these too
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    # real pattern: every 4th layer is full attention; last one must be full
    tc.layer_types = ["linear_attention"] * (layers - 1) + ["full_attention"]
    try:
        tc.mtp_num_hidden_layers = None      # drop the MTP head (not used here)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    model.config.use_cache = False
    return model


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    tok = StubTok()
    hidden = 64
    model = _tiny_qwen(hidden=hidden)
    proj = gp.HourglassProjector(gp.VISION_DIM, hidden)
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3)
    yield tok, model, proj, opt


def _fake_batch(tok, B=3, n_vis=(3, 5, 2)):
    items = [{"vis": torch.randn(nv, gp.VISION_DIM),
              "user": f"prompt {i}", "assistant": f"answer {i}", "g": "t"}
             for i, nv in enumerate(n_vis)]
    return gp.make_collate(tok, tok.pad_token_id, max_len=64)(items)


# ------------------------------- tests ---------------------------------------


def test_projector_param_count_formula():
    # LN(vision) + Linear(vision -> 2*llm) + GELU(0) + Linear(2*llm -> llm)
    d, h = gp.VISION_DIM, 128
    p = gp.HourglassProjector(d, h)
    expected = 2 * d + (d * 2 * h + 2 * h) + (2 * h * h + h)
    assert sum(q.numel() for q in p.parameters()) == expected


def test_end_to_end_five_steps(setup):
    tok, model, proj, opt = setup
    device = "cpu"
    losses = []
    for _ in range(5):
        batch = _fake_batch(tok)
        out = gp.train_step(model, proj, opt, batch, device)
        assert out["finite"], "loss must be finite"
        losses.append(out["loss"])
        assert out["gnorm"] > 0, "grads must reach the projector"
    assert max(losses) - min(losses) > 0, \
        f"loss must vary across steps, got {losses}"


def test_shapes_flow_end_to_end(setup):
    tok, model, proj, _ = setup
    batch = _fake_batch(tok)
    inp = gp.embeds_for(model, batch, proj, "cpu")
    H = model.get_input_embeddings().weight.shape[1]
    assert set(inp) == {"inputs_embeds", "attention_mask", "labels"}
    assert inp["inputs_embeds"].shape == batch["input_ids"].shape + (H,)
    out = model(**inp)
    assert torch.isfinite(out.loss)
    # visual span of row 0 must equal proj(vis[:n_vis]) — splice landed
    nv0 = int(batch["n_vis"][0])
    pv = proj(batch["vis"][:1, :nv0])
    assert torch.allclose(inp["inputs_embeds"][0, 1:1 + nv0], pv, atol=1e-4)
    # BOS position untouched by the splice
    bos_ref = model.get_input_embeddings()(
        torch.tensor([[tok.bos_token_id]]))[0, 0]
    assert torch.allclose(inp["inputs_embeds"][0, 0], bos_ref, atol=1e-6)
    # image span excluded from loss
    assert (inp["labels"][0, 1:1 + nv0] == -100).all()
    # no input_ids escape to the model call
    assert "input_ids" not in inp


def test_grads_reach_projector_through_frozen_backbone(setup):
    tok, model, proj, _ = setup
    for p in proj.parameters():
        p.grad = None
    batch = _fake_batch(tok)
    inp = gp.embeds_for(model, batch, proj, "cpu")
    model(**inp).loss.backward()
    gsum = sum(p.grad.abs().sum().item() for p in proj.parameters() if p.grad is not None)
    assert gsum > 0, "backward through frozen backbone must reach the projector"


def test_collate_invariants_match_suite_pins(setup):
    tok, _, _, _ = setup
    batch = _fake_batch(tok)
    gp.check_collate_invariants(batch, tok)          # must not raise
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    nv = int(batch["n_vis"][0])
    assert ids[0] == tok.bos_token_id
    sup = [i for i, lab in enumerate(labels) if lab != -100]
    assert len(sup) == len(tok("answer 0")["input_ids"]) + 1   # answer tokens + EOS
    assert all(i > 1 + nv for i in sup)
    # trimming keeps the answer intact when the user prompt overflows budget
    long_item = [{"vis": torch.randn(2, gp.VISION_DIM), "user": "u" * 500,
                  "assistant": "keep me", "g": "t"}]
    b2 = gp.make_collate(tok, tok.pad_token_id, max_len=64)(long_item)
    ids2 = b2["input_ids"][0].tolist()
    eos_pos = ids2.index(tok.eos_token_id)
    a = tok("keep me")["input_ids"]
    assert ids2[eos_pos - len(a):eos_pos] == a


def test_resume_artifacts_round_trip(tmp_path, setup):
    tok, model, proj, opt = setup
    path = tmp_path / "latest.safetensors"
    gp.save_projector(proj, opt, 7, str(path))
    assert (tmp_path / "latest.opt.pt").exists()

    from safetensors.torch import load_file
    sd = load_file(str(path))
    ref = {k: v.detach().cpu() for k, v in proj.state_dict().items()}
    assert set(sd) == set(ref)
    for k in sd:
        assert torch.equal(sd[k], ref[k])

    st = torch.load(tmp_path / "latest.opt.pt", map_location="cpu", weights_only=False)
    assert st["step"] == 7
    opt2 = torch.optim.AdamW(proj.parameters(), lr=gp.LR)
    opt2.load_state_dict(st["opt"])


def test_lr_schedule_warmup_then_cosine():
    lrs = [gp.lr_at(s, 1000) for s in range(1, 1001)]
    assert lrs[0] < lrs[50] < lrs[99] <= gp.LR + 1e-9     # warmup ramp to peak at step 100
    assert abs(lrs[99] - gp.LR) < 1e-9
    assert lrs[-1] == pytest.approx(0.1 * gp.LR, rel=1e-6)  # cosine floor = 10% peak
    assert all(a >= b - 1e-12 for a, b in zip(lrs[99:], lrs[100:]))  # monotone DECAY after warmup


def test_monitor_plateau_banner_and_spike_alert(capsys):
    m = gp.ProbeMonitor()
    # flat plateau for > window steps
    step = 0
    while step < gp.PLATEAU_WINDOW + 200:
        step += 1
        m.update(step, 7.0 + (step % 3) * 0.01, step * 8)
    captured = capsys.readouterr().out
    assert "FLAT LOSS IS EXPECTED" in captured
    assert m.n_banners >= 1
    # collapse detector fires on a cliff below half the window floor
    for _ in range(gp.PLATEAU_CHECK_EVERY + 1):
        step += 1
        m.update(step, 0.5, step * 8)
    assert m.collapse_step is not None
    # spike alert on a >2x EMA jump
    for _ in range(60):
        step += 1
        m.update(step, 20.0, step * 8)
    captured = capsys.readouterr().out
    assert "SPIKE-ALERT" in captured or m.n_alerts >= 1


def test_render_curves_smoke(tmp_path):
    recs = [{"step": s, "loss": 8.0 - s * 0.01, "ema_loss": 7.9 - s * 0.008,
             "lr": 5e-4, "gnorm": 1.0, "samples_seen": s * 8}
            for s in range(1, 60)]
    out = tmp_path / "curves.png"
    assert gp.render_curves(recs, str(out))
    assert out.stat().st_size > 1000
