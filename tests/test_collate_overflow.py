"""Overflow guard: make_collate must never emit L > max_len, and must not
starve the text to achieve it.

Live data has n_vis up to 16653 with max_len=4096. The first guard trimmed the
visual span only, which left the *text* budget at 2 tokens — so a huge-n_vis
row crushed its answer to 1-2 tokens and its question to 1 (1,986 rows in the
manifest, 27.7% of all answer words). The budget is now answer > question >
image: the image is the elastic resource (see tests/test_collate_budget.py).

Hermetic: stub tokenizer + tiny vision_dim, no HF. Run:
    python -m pytest tests/test_collate_overflow.py -q
"""

import torch

from vision_adapter.core import make_collate


class StubTok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        ids = [(ord(c) % 900) + 10 for c in text]
        return {"input_ids": ids or [10]}


def _item(n_vis, user="hi", assistant="ans"):
    return {
        "vis": torch.randn(n_vis, 8),
        "user": user,
        "assistant": assistant,
        "g": "test",
    }


def test_overflow_row_capped_at_max_len():
    tok = StubTok()
    batch = make_collate(tok, tok.pad_token_id, max_len=64, vision_dim=8)(
        [_item(n_vis=100)]
    )
    B, L = batch["input_ids"].shape
    assert B == 1
    assert L <= 64, f"overflow row must fit max_len, got L={L}"
    # n_vis contract follows the kept span (splice + positions use it)
    kept = int(batch["n_vis"][0])
    assert kept <= 60, f"visual span must be trimmed to fit, kept={kept}"
    # answer keeps priority: the WHOLE answer survives, EOS present
    ids = batch["input_ids"][0].tolist()
    a = tok("ans")["input_ids"]
    eos_pos = ids.index(tok.eos_token_id)
    assert ids[eos_pos - len(a) : eos_pos] == a, "the answer must not be truncated"
    # labels stay consistent: supervised span is answer+EOS within L
    labels = batch["labels"][0].tolist()
    sup = [i for i, lab in enumerate(labels) if lab != -100]
    assert sup and sup[-1] == eos_pos
    assert batch["attention_mask"][0, : eos_pos + 1].all()


def test_fitting_rows_unchanged_by_guard():
    tok = StubTok()
    batch = make_collate(tok, tok.pad_token_id, max_len=64, vision_dim=8)(
        [_item(n_vis=3)]
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
