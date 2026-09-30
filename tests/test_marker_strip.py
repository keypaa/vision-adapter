"""Marker strip: Sero textual image blocks must not reach the tokenizer.

Live agentic rows start with '|begin_of_image|' + 128x'|image|' +
'|end_of_image|' (265 Qwen tokens) while make_collate separately injects the
real visual embeddings -> double representation. The strip removes the block
and keeps instruction + previous actions. Hermetic. Run:
    python -m pytest tests/test_marker_strip.py -q
"""

import torch

from vision_adapter.core import make_collate, strip_text_image_markers


class StubTok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        ids = [(ord(c) % 900) + 10 for c in text]
        return {"input_ids": ids or [10]}


BLOCK = "|begin_of_image|" + "|image|" * 128 + "|end_of_image|"


def test_strip_removes_block_keeps_text():
    assert strip_text_image_markers(BLOCK + "Instruction: click X") == (
        "Instruction: click X"
    )
    assert strip_text_image_markers("no markers here") == "no markers here"
    assert strip_text_image_markers(BLOCK) == ""


def test_collate_user_span_has_no_marker_tokens():
    tok = StubTok()
    user = BLOCK + "click here"
    item = {
        "vis": torch.randn(3, 8),
        "user": user,
        "assistant": "done",
        "g": "agentic",
    }
    batch = make_collate(tok, tok.pad_token_id, max_len=512, vision_dim=8)([item])
    ids = batch["input_ids"][0].tolist()
    want_u = tok("click here")["input_ids"]
    want_a = tok("done")["input_ids"]
    # [BOS][img x3][user][answer][EOS] with the stripped user text
    assert ids[1:4] == [0, 0, 0]
    assert ids[4 : 4 + len(want_u)] == want_u
    assert ids[4 + len(want_u) : 4 + len(want_u) + len(want_a)] == want_a
    assert ids[4 + len(want_u) + len(want_a)] == tok.eos_token_id
    # the 265-token marker block frees its whole budget
    assert len(ids) < 100
