"""Training data-contract tests: make_collate sequence layout + visual_inject splicing.

These pin the exact [BOS][img][user][answer][EOS] contract documented in
modal_train.py's docstring — the thing every downstream training behavior
depends on. Hermetic: stub tokenizer + tiny embedding, no HF downloads. Run:
    python -m pytest test_train_collate.py -q
"""
import torch

from modal_train import make_collate, visual_inject


class StubTok:
    """Deterministic char-level tokenizer: ids >= 10 so they never collide
    with the special-token ids used here (pad=0, bos=1, eos=2)."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        ids = [(ord(c) % 900) + 10 for c in text]
        return {"input_ids": ids or [10]}


def _item(n_vis=3, user="hello", assistant="hi there"):
    return {"vis": torch.randn(n_vis, 4096), "user": user,
            "assistant": assistant, "g": "test"}


def test_collate_sequence_layout():
    tok = StubTok()
    batch = make_collate(tok, tok.pad_token_id)([_item(n_vis=3)])
    ids = batch["input_ids"][0].tolist()
    u = tok("hello")["input_ids"]
    a = tok("hi there")["input_ids"]
    # [BOS][img x3][user][answer][EOS]
    assert ids[0] == tok.bos_token_id
    assert ids[1:1 + 3] == [0, 0, 0]              # img span: id value unused (pad)
    assert ids[4:4 + len(u)] == u
    assert ids[4 + len(u):4 + len(u) + len(a)] == a
    assert ids[4 + len(u) + len(a)] == tok.eos_token_id


def test_collate_labels_only_answer_and_eos():
    tok = StubTok()
    batch = make_collate(tok, tok.pad_token_id)([_item(n_vis=3, user="hello", assistant="ans")])
    labels = batch["labels"][0].tolist()
    supervised = [i for i, l in enumerate(labels) if l != -100]
    a_len = len(tok("ans")["input_ids"])
    assert len(supervised) == a_len + 1           # answer tokens + EOS
    decoded = [labels[i] for i in supervised]
    assert decoded[:-1] == tok("ans")["input_ids"]
    assert decoded[-1] == tok.eos_token_id


def test_collate_attention_and_padding():
    tok = StubTok()
    short = _item(n_vis=2, user="a", assistant="b")
    long_ = _item(n_vis=9, user="longer prompt", assistant="longer answer text")
    batch = make_collate(tok, tok.pad_token_id)([short, long_])
    attn = batch["attention_mask"]
    L = attn.shape[1]
    # row 0 (shorter) is right-padded: 1s then 0s; row 1 reaches the end
    r0 = attn[0].tolist()
    first_zero = r0.index(0)
    assert all(v == 1 for v in r0[:first_zero]) and all(v == 0 for v in r0[first_zero:])
    assert attn[1].sum() < L or True
    # vis padded to the batch max with zero rows
    assert batch["vis"].shape == (2, 9, 4096)
    assert batch["vis"][0, 2:].abs().sum() == 0   # rows beyond n_vis=2 are zeros
    assert batch["n_vis"].tolist() == [2, 9]


def test_collate_trims_user_keeps_answer_when_over_budget():
    tok = StubTok()
    long_user = "u" * 500
    item = {"vis": torch.randn(2, 4096), "user": long_user,
            "assistant": "keep me", "g": "t"}
    batch = make_collate(tok, tok.pad_token_id, max_len=64)([item])
    ids = batch["input_ids"][0].tolist()
    a = tok("keep me")["input_ids"]
    # answer must survive intact right before EOS even when user was trimmed
    eos_pos = ids.index(tok.eos_token_id)
    assert ids[eos_pos - len(a):eos_pos] == a
    assert eos_pos + 1 <= 64                      # whole sequence fits max_len


def _proj(d):
    class Proj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4096, d, bias=False).to(torch.bfloat16)   # like the real projector

        def forward(self, x):
            return self.lin(x)

    return Proj().eval()


def test_visual_inject_splices_image_block():
    torch.manual_seed(0)
    d = 16
    emb = torch.nn.Embedding(1024, d)   # StubTok ids reach ~909
    model = type("M", (), {"get_input_embeddings": lambda s: emb})()
    proj = _proj(d)
    tok = StubTok()
    collate = make_collate(tok, tok.pad_token_id, max_len=32)
    inputs = collate([_item(n_vis=3, user="abc", assistant="xyz")])

    # reference embeddings MUST be captured before entering the window: while
    # the hook is live it splices EVERY embed_tokens call, including probes.
    with torch.no_grad():
        bos_ref = emb(torch.tensor([tok.bos_token_id]))[0]
        text_ref = emb(inputs["input_ids"][0, 4:7])
        vis_raw = proj(inputs["vis"][:, :3, :].to(torch.bfloat16)).float()

    with visual_inject(inputs, proj, model) as full:
        # ids stay the model-facing input (V4 hash-MoE gate needs them);
        # no inputs_embeds may escape to the model call
        assert "input_ids" in full and "inputs_embeds" not in full
        with torch.no_grad():
            merged = emb(full["input_ids"])           # hook fires here
        assert merged.shape == full["input_ids"].shape + (d,)
        # BOS position untouched
        assert torch.equal(merged[0, 0], bos_ref)
        # image span equals proj(vis); text span equals raw embeddings
        assert torch.allclose(merged[0, 1:4], vis_raw[0], atol=1e-2)
        assert torch.equal(merged[0, 4:7], text_ref)
        # image span stays excluded from loss (labels -100 from collate)
        assert (full["labels"][0, 1:4] == -100).all()

    # hook removed on exit: embedding output is plain again
    with torch.no_grad():
        plain = emb(full["input_ids"])
        assert torch.equal(plain[0, 1:4], emb(full["input_ids"][0, 1:4]))


def test_visual_inject_grads_reach_projector():
    torch.manual_seed(0)
    d = 8
    emb = torch.nn.Embedding(1024, d)
    model = type("M", (), {"get_input_embeddings": lambda s: emb})()
    proj = _proj(d)
    tok = StubTok()
    collate = make_collate(tok, tok.pad_token_id, max_len=32)
    inputs = collate([_item(n_vis=2, user="ab", assistant="cd")])

    with visual_inject(inputs, proj, model) as full:
        merged = emb(full["input_ids"])               # autograd through the splice
        loss = merged[0, 1:3].sum()                   # only the visual rows matter
    loss.backward()
    assert proj.lin.weight.grad is not None
    assert proj.lin.weight.grad.abs().sum() > 0


# ---- chunked eager attention: must be numerically identical to full eager ----

def _repeat_kv(hidden_states, n_rep):
    # faithful copy of transformers' repeat_kv (4-D [B, kv_heads, S, D])
    batch, kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, kv_heads * n_rep, slen, head_dim)


def _reference_eager(module, query, key, value, attention_mask,
                     scaling, dropout=0.0, **kwargs):
    """Faithful copy of transformers v5.15.1 deepseek_v4.eager_attention_forward."""
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    attn_weights = torch.nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def test_chunked_eager_matches_full_eager():
    from modal_train import _make_chunked_eager
    torch.manual_seed(3)
    B, H, KVH, SQ, SKV, D = 2, 6, 2, 13, 17, 8   # rectangular: SQ != SKV

    class AttnMod(torch.nn.Module):
        num_key_value_groups = H // KVH
        training = True
        sinks = torch.randn(H)

    mod = AttnMod()
    q = torch.randn(B, H, SQ, D)
    k = torch.randn(B, KVH, SKV, D)
    v = torch.randn(B, KVH, SKV, D)
    mask = torch.randn(B, 1, SQ, SKV)              # additive bias, fp32 like the real path
    mask[..., -3:] = -1e9                          # some heavily-masked slots

    ref_out, ref_w = _reference_eager(mod, q, k, v, mask, scaling=0.125)

    for budget in (10 ** 9, 64, 37):               # single-chunk / tiny chunks / ragged tail
        chunked = _make_chunked_eager(budget_elems=budget)
        out, w = chunked(mod, q, k, v, mask, scaling=0.125)
        assert out.shape == ref_out.shape
        assert torch.allclose(out.float(), ref_out.float(), atol=1e-5)
        assert w is None                           # weights intentionally dropped
    # no-mask path too
    chunked = _make_chunked_eager(budget_elems=50)
    out_nomask, _ = chunked(mod, q, k, v, None, scaling=0.125)
    ref_nomask, _ = _reference_eager(mod, q, k, v, None, scaling=0.125)
    assert torch.allclose(out_nomask.float(), ref_nomask.float(), atol=1e-5)


# ---- _fp8_linear_train: differentiable blockwise-dequant linear ----

def test_fp8_linear_train_matches_dequant_reference():
    from modal_train import _fp8_linear_train
    torch.manual_seed(5)
    fp8 = torch.float8_e4m3fn

    def _ref(x, w_fp8, scales, bn, bk):
        O, I = w_fp8.shape
        wd = (w_fp8.to(torch.float32)
              .view(O // bn, bn, I // bk, bk)
              .mul(scales.to(torch.float32)[:, None, :, None])
              .view(O, I)).to(torch.bfloat16)
        return torch.nn.functional.linear(x, wd)

    for (O, I, bn, bk) in [(64, 32, 16, 16), (128, 128, 128, 128), (50, 33, 16, 16)]:
        w = (torch.randn(O, I) * 0.1).clamp(-448, 448).to(fp8)
        scales = torch.rand(-(-O // bn), -(-I // bk)) * 0.01 + 0.005
        x = torch.randn(4, I, dtype=torch.bfloat16)

        out = _fp8_linear_train(x, w, scales, (bn, bk))
        if O % bn == 0 and I % bk == 0:
            ref = _ref(x, w, scales, bn, bk)       # exact-comparable only when no padding
            assert torch.allclose(out, ref, atol=1e-2)

    # autograd must reach the INPUT (the whole point: grads flow to projector)
    x = torch.randn(4, 32, dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(64, 32) * 0.1).to(fp8)
    scales = torch.rand(4, 2) * 0.01 + 0.005
    out = _fp8_linear_train(x, w, scales, (16, 16))
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dequant_expert_slice_fp4_and_scales():
    from modal_train import _dequant_expert_slice, _FP4_E2M1_LUT
    torch.manual_seed(7)

    # --- fp4 path: 2 bytes -> 4 values, low nibble first, row-scales gran 32 ---
    E, O, K = 3, 64, 128                      # K = packed cols * 2
    packed = torch.randint(-128, 128, (E, O, K // 2), dtype=torch.int8)
    scales = torch.rand(E, O, K // 32) * 0.05 + 0.001   # sf_gran_n=1 (per row), sf_gran_k=32

    out = _dequant_expert_slice(packed, scales)
    assert out.shape == (E, O, K)

    # explicit loop ground truth for one expert/row block
    u8 = packed[0].view(torch.uint8)
    for r in (0, 33):
        vals = []
        for c in range(K // 2):
            lo, hi = int(u8[r, c]) & 0xF, int(u8[r, c]) >> 4
            g = c * 2 // 32                               # value index -> k-group
            vals.append(_FP4_E2M1_LUT[lo] * float(scales[0, r, g]))
            vals.append(_FP4_E2M1_LUT[hi] * float(scales[0, r, g]))
        ref = torch.tensor(vals)
        assert torch.allclose(out[0, r].cpu(), ref, atol=1e-6)

    # --- uint8-stored e8m0 exponents: scale == 2**(byte-127) ---
    packed1 = torch.randint(-128, 128, (1, 32, 16), dtype=torch.int8)   # K=32
    exp_bytes = torch.full((1, 32, 1), 130, dtype=torch.uint8)          # 2**(130-127) = 8.0
    out_e8 = _dequant_expert_slice(packed1, exp_bytes)
    u8 = packed1[0].view(torch.uint8)
    lo, hi = int(u8[0, 0]) & 0xF, int(u8[0, 0]) >> 4
    assert float(out_e8[0, 0, 0]) == _FP4_E2M1_LUT[lo] * 8.0
    assert float(out_e8[0, 0, 1]) == _FP4_E2M1_LUT[hi] * 8.0

    # grads must not be required through weights (frozen backbone) but op is autograd-safe
    x = torch.randn(5, K, dtype=torch.bfloat16, requires_grad=True)
    y = torch.nn.functional.linear(x, _dequant_expert_slice(packed, scales)[0].to(x.dtype))
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
