"""True MoonViT geometry: measured (gh, gw) must beat the synthetic stand-in.

Audit 2026-09-30: grid_for_nvis(n_vis) factors a token count into a squarest
even grid, which INVERTS orientation — a 56x26 portrait (n_vis=364) was fed to
mRoPE as 28x52 landscape (aspect error x3.0 median, 200/200 rows off). The
real grid is a deterministic function of the image dims (verified 400/400 on
the live corpus), so batches carrying grid_thw must use it.

Run: python -m pytest tests/test_true_geometry.py -q
"""
import torch

from vision_adapter.core import (
    grid_for_nvis,
    train_position_ids,
    vision_position_ids,
)


def _batch(n_vis, grid_thw=None, L=None):
    L = L or (2 + n_vis)
    return {
        "input_ids": torch.zeros(1, L, dtype=torch.long),
        "n_vis": torch.tensor([n_vis], dtype=torch.long),
        "grid_thw": grid_thw,
    }


def test_measured_grid_drives_mrope_positions():
    """A 56x26 portrait grid (28x13 after the 2x2 merge) positions row-major."""
    batch = _batch(364, grid_thw=[[1, 56, 26]])
    pos = train_position_ids(batch)
    # rows: 1 = temporal (t=1 everywhere), 2 = height, 3 = width (offset by start=1)
    assert all(int(v) == 1 for v in pos[1, 0, 1:365])
    # width sweeps 1..13 within an LLM row, then restarts
    assert [int(v) for v in pos[3, 0, 1:14]] == list(range(1, 14))
    assert [int(v) for v in pos[3, 0, 14:27]] == list(range(1, 14))
    # height advances only after 13 columns -> 28 distinct rows over 364 placeholders
    assert all(int(v) == 1 for v in pos[2, 0, 1:14])
    assert all(int(v) == 2 for v in pos[2, 0, 14:27])
    assert int(pos[2, 0, 364]) == 28


def test_grid_from_dims_reproduces_a_measured_portrait_grid():
    """364x784 px (live aitw_000000.jpg) must yield the stored [1,56,26]/n_vis=364."""
    from vision_adapter.core import grid_from_dims

    g = grid_from_dims(364, 784)
    assert g.tolist() == [1, 56, 26]
    assert int(g[0] * g[1] * g[2]) // 4 == 364
    # the fallback the audit rejected: same n_vis, landscape
    assert grid_for_nvis(364, 2).tolist() == [1, 28, 52]


def test_measured_grid_differs_from_synthetic():
    """The regression itself: synthetic and measured must not agree here."""
    measured = torch.tensor([1, 56, 26])
    synthetic = grid_for_nvis(364)
    assert measured.tolist() != synthetic.tolist()
    assert int(torch.prod(measured)) // 4 == 364


def test_missing_grid_falls_back_to_synthetic():
    batch = _batch(4, grid_thw=None)
    pos = train_position_ids(batch)
    expected = vision_position_ids(1, grid_for_nvis(4))
    assert torch.equal(pos[1:, 0, 1:5], expected)


def test_placeholder_count_mismatch_is_rejected():
    """A grid that disagrees with n_vis is a data bug — never silently used."""
    import pytest

    with pytest.raises(ValueError, match="mismatch"):
        train_position_ids(_batch(364, grid_thw=[[1, 56, 28]]))


def test_per_row_grids_are_independent():
    """Two rows with identical n_vis but different aspect ratios differ."""
    batch = {
        "input_ids": torch.zeros(2, 6, dtype=torch.long),
        "n_vis": torch.tensor([4, 4], dtype=torch.long),
        "grid_thw": [[1, 4, 4], [1, 2, 8]],
    }
    pos = train_position_ids(batch)
    assert not torch.equal(pos[3, 0, 1:5], pos[3, 1, 1:5])


def test_collate_preserves_grid_thw_per_row():
    """The measured grid must survive collate, else training can't see it."""
    from vision_adapter.core import make_collate

    class StubTok:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [(ord(c) % 900) + 10 for c in text] or [10]}

    tok = StubTok()
    items = [
        {
            "vis": torch.randn(4, 8),
            "user": "u",
            "assistant": "a",
            "g": "t",
            "grid_thw": [1, 4, 8],
        },
        {"vis": torch.randn(4, 8), "user": "u", "assistant": "a", "g": "t"},
    ]
    batch = make_collate(tok, tok.pad_token_id, max_len=64, vision_dim=8)(items)
    assert "grid_thw" in batch
    # row 0 measured, row 1 left empty for the synthetic fallback
    assert batch["grid_thw"][0] == [1, 4, 8]
    assert not batch["grid_thw"][1]


def test_mixed_batch_measures_only_rows_with_a_grid():
    """A half-measured batch is the real regime: measure what we have."""
    batch = {
        "input_ids": torch.zeros(2, 10, dtype=torch.long),
        "n_vis": torch.tensor([8, 8], dtype=torch.long),
        "grid_thw": [[1, 4, 8], None],   # 4x8/4 = 8 placeholders
    }
    pos = train_position_ids(batch)
    # row 0 measured: [1,4,8] -> 2 rows x 4 cols after the merge, w fastest
    assert [int(v) for v in pos[3, 0, 1:9]] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert [int(v) for v in pos[2, 0, 1:9]] == [1, 1, 1, 1, 2, 2, 2, 2]
    # row 1 unmeasured: synthetic for n_vis=8 is the same (4, 8) split
    assert pos[3, 1, 1:9].tolist() == [1, 2, 3, 4, 1, 2, 3, 4]


def test_run_card_declares_grid_source():
    """A run on the synthetic stand-in must say so; curves are not comparable."""
    from vision_adapter.config import config_header, default_config

    syn = config_header(default_config(), extra={"run": "t"})
    assert syn["grid_source"] == "synthetic"

    import dataclasses
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "grid.json"
        p.write_text("{}")
        cfg = dataclasses.replace(default_config(), grid_sidecar=str(p))
        meas = config_header(cfg, extra={"run": "t"})
    assert meas["grid_source"] == "measured"
