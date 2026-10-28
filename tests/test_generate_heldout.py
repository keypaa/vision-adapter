import pytest
import torch

from scripts.generate_heldout import (
    plan_for_selected_rows,
    select_agentic_heldout_rows,
    validate_streamed_rows,
)


HELDOUT = "data/emb_0000.parquet"
TRAINED = "data/emb_0005.parquet"


def _row(key, group="agentic", grid=(1, 2, 2), answer="click(start_box=[1, 2])"):
    return {
        "emb": key,
        "g": group,
        "grid_thw": list(grid) if grid is not None else None,
        "user": f"prompt for {key}",
        "assistant": answer,
    }


def test_select_agentic_heldout_rows_is_deterministic_and_in_regime():
    rows = [
        _row("held-0"),
        _row("held-1", group="doc"),
        _row("held-2", grid=(1, 40, 40)),  # 400 merged tokens, over cap
        _row("trained"),
        _row("no-grid", grid=None),
    ]
    index = {
        "held-0": (HELDOUT, 12, 1),
        "held-1": (HELDOUT, 13, 1),
        "held-2": (HELDOUT, 14, 400),
        "trained": (TRAINED, 15, 1),
        "no-grid": (HELDOUT, 16, 1),
    }

    selected, counts = select_agentic_heldout_rows(rows, index, n=5, seed=17)
    repeated, _ = select_agentic_heldout_rows(rows, index, n=5, seed=17)
    assert [r["emb"] for r in selected] == [r["emb"] for r in repeated]
    assert [r["emb"] for r in selected] == ["held-0"]
    assert selected[0]["_stream_row"] == 12
    assert counts["eligible"] == 1
    assert counts["over_token_cap"] == 1
    assert counts["missing_grid"] == 1
    assert counts["not_agentic"] == 1
    assert counts["not_heldout"] == 1


def test_selected_plan_preserves_exact_shard_row_and_measured_grid():
    selected, _ = select_agentic_heldout_rows(
        [_row("b"), _row("a")],
        {"a": (HELDOUT, 20, 1), "b": (HELDOUT, 10, 1)},
        n=2,
    )
    plan = plan_for_selected_rows(selected)
    assert [row["_row"] for row in plan[HELDOUT]] == [10, 20]
    assert all(row["grid_thw"] == [1, 2, 2] for row in plan[HELDOUT])
    assert {row["emb"] for row in plan[HELDOUT]} == {"a", "b"}


def test_validation_selection_can_use_any_shard_and_token_floor():
    rows = [_row("small"), _row("typical", grid=(1, 40, 38))]
    index = {
        "small": (TRAINED, 1, 1),
        "typical": ("data/emb_0020.parquet", 2, 380),
    }
    selected, counts = select_agentic_heldout_rows(
        rows, index, n=5, seed=0, min_n_vis=300, max_n_vis=380,
        allowed_shards=None,
    )
    assert [row["emb"] for row in selected] == ["typical"]
    assert counts["under_token_floor"] == 1
    assert counts["eligible"] == 1


def test_grid_disagreement_with_index_is_fatal():
    with pytest.raises(ValueError, match="grid/token mismatch"):
        select_agentic_heldout_rows(
            [_row("bad")], {"bad": (HELDOUT, 0, 2)}, n=1
        )


def test_validate_streamed_rows_requires_real_expected_embeddings():
    selected, _ = select_agentic_heldout_rows(
        [_row("k")], {"k": (HELDOUT, 0, 1)}, n=1
    )
    streamed = [{
        "emb": "k",
        "g": "agentic",
        "user": selected[0]["user"],
        "assistant": selected[0]["assistant"],
        "grid_thw": [1, 2, 2],
        "vis": torch.zeros(1, 4096),
    }]
    assert validate_streamed_rows(selected, streamed)[0]["emb"] == "k"
    with pytest.raises(RuntimeError, match="did not return"):
        validate_streamed_rows(selected, [])


def test_validate_streamed_rows_rejects_missing_or_changed_grid():
    selected, _ = select_agentic_heldout_rows(
        [_row("k")], {"k": (HELDOUT, 0, 1)}, n=1
    )
    streamed = [{
        "emb": "k",
        "g": "agentic",
        "user": selected[0]["user"],
        "assistant": selected[0]["assistant"],
        "grid_thw": None,
        "vis": torch.zeros(1, 4096),
    }]
    with pytest.raises(RuntimeError, match="lost/changed measured grid"):
        validate_streamed_rows(selected, streamed)
