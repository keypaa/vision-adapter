"""Grid sidecar: emb_key -> measured (1, gh, gw), built without re-running the ViT.

The corpus parquet already stores the true image dims, and preprocess is
deterministic, so the MoonViT grid is recoverable (verified 400/400 against
the n_vis in the embedding key index). The sidecar persists it so training
can position mRoPE on measured geometry instead of the synthetic stand-in.

Run: python -m pytest tests/test_grid_sidecar.py -q
"""
import json

from vision_adapter.grid_sidecar import GridSidecar, build_sidecar_rows


class _FakeBatch:
    def __init__(self, d):
        self._d = d

    def column(self, name):
        class _C:
            def __init__(self, v):
                self._v = v

            def to_pylist(self):
                return self._v

        return _C(self._d[name])


def test_rows_derive_grid_from_image_dims():
    batches = [
        _FakeBatch(
            {
                "filename": ["agentic/aitw_000000.jpg", "cauldron/x.png"],
                "image": [b"fake-bytes-a", b"fake-bytes-b"],
                "size": [10, 20],
            }
        )
    ]
    sizes = {("agentic", "aitw_000000.jpg"): (364, 784)}
    rows = build_sidecar_rows(
        batches,
        key_fn=lambda g, b: f"embeddings/{g}-{b}.pt",
        dims_fn=lambda g, b: sizes.get((g, b)),
    )
    assert [r["emb"] for r in rows] == ["embeddings/agentic-aitw_000000.jpg.pt"]
    # 364x784 px -> the measured portrait grid verified against the key index
    assert rows[0]["grid_thw"] == [1, 56, 26]


def test_missing_dims_produce_no_row():
    """A file whose dims we could not read is omitted, never guessed."""
    batches = [_FakeBatch({"filename": ["a.png"], "image": [b"x"], "size": [1]})]
    rows = build_sidecar_rows(
        batches, key_fn=lambda *_: "embeddings/aa.pt", dims_fn=lambda *_: None
    )
    assert rows == []


def test_sidecar_roundtrip(tmp_path):
    p = tmp_path / "grid_sidecar.json"
    p.write_text(json.dumps({"embeddings/aa.pt": [1, 56, 26]}))
    sc = GridSidecar(p)
    assert sc.get("embeddings/aa.pt") == [1, 56, 26]
    assert sc.get("embeddings/missing.pt") is None


def test_sidecar_absent_file_is_empty(tmp_path):
    sc = GridSidecar(tmp_path / "nope.json")
    assert sc.get("embeddings/aa.pt") is None
    assert len(sc) == 0


def test_sidecar_grid_for_batch_falls_back_per_row():
    sc = GridSidecar(None)
    sc.data["embeddings/aa.pt"] = [1, 56, 26]
    n_vis = [364, 4]
    embs = ["embeddings/aa.pt", "embeddings/bb.pt"]
    grids, sources = sc.grids_for(embs, n_vis)
    assert grids[0] == [1, 56, 26] and sources[0] == "measured"
    # no entry for bb -> synthetic, and the caller can see it
    assert sources[1] == "synthetic"
    assert grids[1] != grids[0]


def test_dataset_item_carries_measured_grid(tmp_path):
    """A stream item must expose grid_thw so collate/train can use it."""
    import json as _json

    from vision_adapter.data.stream import EmbStreamDataset

    p = tmp_path / "grid.json"
    p.write_text(_json.dumps({"embeddings/aa.pt": [1, 56, 26]}))
    ds = EmbStreamDataset(plan={}, stream_order=[], grid_sidecar=str(p))
    item = ds.attach_grid({"emb": "embeddings/aa.pt", "n_vis": 364})
    assert item["grid_thw"] == [1, 56, 26]
    # unknown emb -> no grid, caller falls back and can label the run synthetic
    assert "grid_thw" not in ds.attach_grid({"emb": "embeddings/zz.pt", "n_vis": 4})


def test_dataset_without_sidecar_leaves_items_untouched():
    from vision_adapter.data.stream import EmbStreamDataset

    ds = EmbStreamDataset(plan={}, stream_order=[])
    row = {"emb": "embeddings/aa.pt", "n_vis": 364}
    assert ds.attach_grid(dict(row)) == row
