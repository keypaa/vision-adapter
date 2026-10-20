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


# ---------------------------------------------------------------- the grid ---
# The gap that let a crash through: the collate may crop the visual span, and
# the mRoPE position path requires a grid whose placeholder count equals the
# number of visual tokens. The crop is therefore row-aligned and the grid is
# cropped with it (core.crop_to_grid).


def _grid_item(n_vis, grid, user="q", assistant="ans"):
    return {"vis": torch.randn(n_vis, 8), "user": user, "assistant": assistant,
            "g": "t", "grid_thw": list(grid)}


def test_measured_grid_stays_consistent_with_a_cropped_span():
    """A row whose image is cropped must still satisfy the grid contract."""
    from vision_adapter.core import placeholder_count, train_position_ids

    tok = StubTok()
    # grid 1x8x2100 patches -> 4x1050 = 4200 merged tokens, cropped to fit 4096
    grid = [1, 8, 2100]
    n_vis = 4200
    batch = make_collate(tok, tok.pad_token_id, max_len=4096, vision_dim=8)(
        [_grid_item(n_vis, grid)]
    )
    kept = int(batch["n_vis"][0])
    assert kept < n_vis, "the image must have been cropped"
    g = batch["grid_thw"][0]
    assert g is not None, "the cropped grid must be reported"
    assert placeholder_count(torch.tensor(g)) == kept, (g, kept, n_vis)
    # and the position code must accept it
    pos = train_position_ids(batch, merge_size=2)
    assert pos.shape[-1] == batch["input_ids"].shape[-1]


def test_crop_is_row_aligned():
    from vision_adapter.core import crop_to_grid

    # a 4x1015 merged grid (8x2030 patches): a crop must land on a row boundary
    kept, grid = crop_to_grid([1, 8, 2030], 4052)
    assert kept % 1015 == 0, kept
    assert kept == 4055 or kept == 3045 or kept == 2029, kept
    assert grid[1] % 2 == 0 and grid[1] // 2 == kept // 1015, (grid, kept)


def test_no_measured_grid_is_fine():
    """Rows without a measured grid fall back to the synthetic one."""
    from vision_adapter.core import train_position_ids

    tok = StubTok()
    batch = make_collate(tok, tok.pad_token_id, max_len=64, vision_dim=8)(
        [_item(n_vis=50, user="q", assistant="ans")]
    )
    assert batch["grid_thw"][0] is None
    pos = train_position_ids(batch, merge_size=2)
    assert pos.shape[-1] == batch["input_ids"].shape[-1]


# ------------------------------------------------------------ O3/O4/O5 ----
# The grid the position code can actually express, and the crops it can
# represent without lying about the geometry.


def test_grid_token_count_matches_the_placeholder_count():
    from vision_adapter.core import grid_token_count, placeholder_count

    for grid in ([1, 8, 2030], [1, 28, 28], [1, 4, 8]):
        assert grid_token_count(grid) == placeholder_count(torch.tensor(grid))


def test_multi_frame_grid_crops_in_whole_frames():
    """t>1: the temporal axis is NOT merged, so the enumeration and the
    placeholder count agree and a head crop is expressible in whole frames."""
    from vision_adapter.core import crop_to_grid, grid_token_count, placeholder_count

    grid = [2, 8, 8]                     # 2 frames x 4 x 4 merged cells = 32 tokens
    assert grid_token_count(grid) == placeholder_count(torch.tensor(grid)) == 32
    kept, cropped = crop_to_grid(grid, 16)
    assert kept == 16, (kept, cropped)
    assert placeholder_count(torch.tensor(cropped)) == kept
    assert grid_token_count(cropped) == kept
    assert cropped[0] == 1, cropped


def test_odd_dims_are_rejected():
    import pytest

    from vision_adapter.core import crop_to_grid

    with pytest.raises(ValueError, match="divisible"):
        crop_to_grid([1, 5, 5], 4)


def test_crop_keeps_the_geometry_exact():
    """The kept tokens must be the FIRST cells of the cropped grid, and the
    cropped grid's count must equal the kept tokens."""
    from vision_adapter.core import (crop_to_grid, grid_token_count, placeholder_count,
                                     vision_position_ids)

    grid = [1, 8, 2100]                     # 4 x 1050 merged cells = 4200 tokens
    kept, cropped = crop_to_grid(grid, 4052)
    assert placeholder_count(torch.tensor(cropped)) == kept
    assert grid_token_count(cropped) == kept
    # the positions of the cropped grid are the first `kept` of the original
    full = vision_position_ids(1, torch.tensor(grid), 2)
    part = vision_position_ids(1, torch.tensor(cropped), 2)
    assert torch.equal(part, full[:, :kept])
    # the crop is whole rows: the leftover budget (< one row) is unused
    assert kept % 1050 == 0
    assert kept <= 4052


def test_a_cropped_row_still_passes_the_position_code():
    from vision_adapter.core import train_position_ids

    tok = StubTok()
    grid = [1, 8, 2100]
    n_vis = 4200
    batch = make_collate(tok, tok.pad_token_id, max_len=4096, vision_dim=8)(
        [_grid_item(n_vis, grid)]
    )
    kept = int(batch["n_vis"][0])
    assert 0 < kept < n_vis
    pos = train_position_ids(batch, merge_size=2)
    assert pos.shape[-1] == batch["input_ids"].shape[-1]


def test_an_image_too_wide_to_crop_keeps_no_visual_tokens():
    """A row whose image cannot be cropped to even ONE merged row (it is wider
    than the whole budget) cannot be represented as a rectangle: the row keeps
    no visual tokens and its text still trains. Needs an image wider than
    ~4,000 merged columns; it does not occur in the corpus, and it must not
    crash the run."""
    from vision_adapter.core import train_position_ids

    tok = StubTok()
    grid = [1, 8, 4060]            # llm_w = 2030 merged cells
    n_vis = 4 * 2030
    batch = make_collate(tok, tok.pad_token_id, max_len=64, vision_dim=8)(
        [_grid_item(n_vis, grid)]
    )
    assert int(batch["n_vis"][0]) == 0
    assert batch["grid_thw"][0] == [1, 0, 4060]
    pos = train_position_ids(batch, merge_size=2)
    assert pos.shape[-1] == batch["input_ids"].shape[-1]
