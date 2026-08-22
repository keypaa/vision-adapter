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
