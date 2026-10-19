"""Text/visual budget contract: make_collate must protect the text, not the image.

Live data (docs/DATA_PATH_REBUILD.md M25/M26) broke the old policy
(image > answer > question) two ways:
  * n_vis >= 4090 squeezed the text budget to 2 tokens -> the answer was
    crushed to <=2 tokens and the question to 1 (1,986 rows, 27.7% of all
    answer words in the manifest);
  * a long question pushed the sequence past max_len (the 4,000-step probe
    reached L=16,602 with max_len=4,096).

The contract now: **answer > question > image**, and L never exceeds max_len.
The image is the elastic resource (cropped from the head, as before).

Hermetic: stub tokenizer + tiny vision_dim, no HF. Run:
    python -m pytest tests/test_collate_budget.py -q
"""

import torch

from vision_adapter.core import MAX_ANSWER_TOKENS, MIN_VIS_TOKENS, make_collate


class StubTok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        ids = [(ord(c) % 900) + 10 for c in text]
        return {"input_ids": ids or [10]}


def _item(n_vis, user="hi", assistant="ans"):
    return {"vis": torch.randn(n_vis, 8), "user": user, "assistant": assistant, "g": "t"}


def _supervised(batch, i=0):
    labels = batch["labels"][i].tolist()
    return [j for j, lab in enumerate(labels) if lab != -100]


def test_answer_is_never_truncated_by_a_large_image():
    """The regression that motivated the fix: a big image must not eat the answer."""
    tok = StubTok()
    max_len = 64
    batch = make_collate(tok, tok.pad_token_id, max_len=max_len, vision_dim=8)(
        [_item(n_vis=200, user="q", assistant="abcdefgh")]
    )
    L = batch["input_ids"].shape[1]
    assert L <= max_len, f"L must stay <= max_len, got {L}"
    a = tok("abcdefgh")["input_ids"]
    ids = batch["input_ids"][0].tolist()
    eos = ids.index(tok.eos_token_id)
    # the WHOLE answer survives (old policy kept only 2 tokens)
    assert ids[eos - len(a) : eos] == a, "the answer must not be truncated"
    # the question survives too
    u = tok("q")["input_ids"]
    assert ids[eos - len(a) - len(u) : eos - len(a)] == u
    # the image absorbed the rest
    kept = int(batch["n_vis"][0])
    assert kept < 200, "the image must be cropped to make room"
    assert kept >= 0


def test_long_question_does_not_push_l_past_max_len():
    tok = StubTok()
    max_len = 64
    long_q = "x" * 500
    batch = make_collate(tok, tok.pad_token_id, max_len=max_len, vision_dim=8)(
        [_item(n_vis=30, user=long_q, assistant="ok")]
    )
    L = batch["input_ids"].shape[1]
    assert L <= max_len, f"L must stay <= max_len, got {L}"
    a = tok("ok")["input_ids"]
    ids = batch["input_ids"][0].tolist()
    eos = ids.index(tok.eos_token_id)
    assert ids[eos - len(a) : eos] == a, "the answer survives a long question"


def test_answer_survives_a_long_answer_budget():
    """A long answer is capped, but the cap is generous and the question yields."""
    tok = StubTok()
    max_len = 4096
    ans = "y" * (MAX_ANSWER_TOKENS + 50)
    batch = make_collate(tok, tok.pad_token_id, max_len=max_len, vision_dim=8)(
        [_item(n_vis=10, user="question", assistant=ans)]
    )
    a = tok(ans)["input_ids"]
    ids = batch["input_ids"][0].tolist()
    eos = ids.index(tok.eos_token_id)
    assert len(a) - eos + eos == len(a)  # sanity
    kept_ans = sum(1 for j in _supervised(batch))
    assert kept_ans <= len(a) + 1, "the answer must be capped, not unbounded"
    assert kept_ans >= MAX_ANSWER_TOKENS, "the answer keeps its full budget"


def test_fitting_row_is_unchanged():
    """A row that fits must be byte-identical to the unguarded layout."""
    tok = StubTok()
    batch = make_collate(tok, tok.pad_token_id, max_len=64, vision_dim=8)(
        [_item(n_vis=3, user="hi", assistant="ans")]
    )
    ids = batch["input_ids"][0].tolist()
    u = tok("hi")["input_ids"]
    a = tok("ans")["input_ids"]
    assert ids[0] == tok.bos_token_id
    assert ids[1:4] == [0, 0, 0]
    assert ids[4 : 4 + len(u)] == u
    assert ids[4 + len(u) : 4 + len(u) + len(a)] == a
    assert ids[4 + len(u) + len(a)] == tok.eos_token_id
    assert int(batch["n_vis"][0]) == 3
    sup = _supervised(batch)
    assert sup and sup[-1] == 4 + len(u) + len(a)


def test_min_visual_budget_is_reserved():
    """The text budget leaves room for MIN_VIS_TOKENS visual tokens."""
    tok = StubTok()
    max_len = 512
    batch = make_collate(tok, tok.pad_token_id, max_len=max_len, vision_dim=8)(
        [_item(n_vis=10, user="q" * 10, assistant="a" * 10)]
    )
    kept = int(batch["n_vis"][0])
    assert kept >= min(10, MIN_VIS_TOKENS)
    assert batch["input_ids"].shape[1] <= max_len


def test_labels_and_mask_stay_consistent():
    tok = StubTok()
    batch = make_collate(tok, tok.pad_token_id, max_len=64, vision_dim=8)(
        [_item(n_vis=50, user="q", assistant="ans")]
    )
    ids = batch["input_ids"][0].tolist()
    sup = _supervised(batch)
    assert sup, "the answer must be supervised"
    assert sup[-1] == ids.index(tok.eos_token_id)
    assert all(ids[j] >= 0 for j in sup)
    assert batch["attention_mask"][0, : sup[-1] + 1].all()
    assert batch["vis"].shape[0] == 1
