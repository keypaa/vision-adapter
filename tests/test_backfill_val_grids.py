import pytest

from scripts.backfill_val_grids import (
    _emb_key, backfill_rows, find_images, image_sizes_from_paths,
    select_val_rows_by_n_vis,
)


HELDOUT = "data/emb_0020.parquet"


def test_validation_sampling_uses_stored_n_vis_and_is_deterministic():
    rows = [
        {"emb": "low", "g": "agentic"},
        {"emb": "mid-a", "g": "agentic"},
        {"emb": "mid-b", "g": "agentic"},
        {"emb": "doc", "g": "doc"},
    ]
    index = {
        "low": (HELDOUT, 0, 35),
        "mid-a": (HELDOUT, 1, 320),
        "mid-b": (HELDOUT, 2, 364),
        "doc": (HELDOUT, 3, 350),
    }
    chosen, counts = select_val_rows_by_n_vis(
        rows, index, n=1, seed=7, min_n_vis=300, max_n_vis=380,
    )
    repeated, _ = select_val_rows_by_n_vis(
        rows, index, n=1, seed=7, min_n_vis=300, max_n_vis=380,
    )
    assert [row["emb"] for row in chosen] == [row["emb"] for row in repeated]
    assert chosen[0]["emb"] in {"mid-a", "mid-b"}
    assert counts["eligible"] == 2
    assert counts["selected"] == 1


def test_backfill_grid_from_original_image_and_verify_n_vis(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "images"
    group_dir = image_dir / "agentic"
    group_dir.mkdir(parents=True)
    image_path = group_dir / "sample.png"
    Image.new("RGB", (28, 28)).save(image_path)
    key = _emb_key("agentic/sample.png")
    rows = [{"emb": key, "g": "agentic", "user": "u", "assistant": "a"}]

    paths = find_images(image_dir, {key})
    out = backfill_rows(rows, image_sizes_from_paths(paths), {key: (HELDOUT, 12, 1)})

    assert out[0]["grid_thw"] == [1, 2, 2]
    assert rows[0].get("grid_thw") is None  # source rows are not mutated


def test_backfill_fails_if_image_key_or_n_vis_does_not_match(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "images"
    group_dir = image_dir / "agentic"
    group_dir.mkdir(parents=True)
    Image.new("RGB", (28, 28)).save(group_dir / "sample.png")
    key = _emb_key("agentic/sample.png")
    rows = [{"emb": key, "g": "agentic"}]
    paths = find_images(image_dir, {key})
    sizes = image_sizes_from_paths(paths)

    with pytest.raises(ValueError, match="image-derived grid mismatch"):
        backfill_rows(rows, sizes, {key: (HELDOUT, 12, 2)})
    with pytest.raises(FileNotFoundError, match="could not resolve"):
        backfill_rows(rows, {}, {key: (HELDOUT, 12, 1)})
