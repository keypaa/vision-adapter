"""disjoint_val_rows: val rows must not share emb keys with train.

The live train_manifest_val.jsonl overlaps train by 47% of emb keys, making
val_loss optimistic. The rebuild keeps val rows whose emb is absent from
train (order preserved). Hermetic. Run:
    python -m pytest tests/test_disjoint_val.py -q
"""

from vision_adapter.manifest import disjoint_val_rows


def _row(emb, g="doc"):
    return {"emb": emb, "user": "u", "assistant": "a", "g": g}


def test_drops_val_rows_seen_in_train():
    train = [_row("embeddings/aa.pt"), _row("embeddings/bb.pt")]
    val = [_row("embeddings/bb.pt"), _row("embeddings/cc.pt")]
    kept = disjoint_val_rows(train, val)
    assert [r["emb"] for r in kept] == ["embeddings/cc.pt"]


def test_ignores_header_rows_and_preserves_order():
    train = [
        {"type": "manifest_header", "emb": "embeddings/zz.pt"},
        _row("embeddings/aa.pt"),
    ]
    val = [
        _row("embeddings/mm.pt", g="conv"),
        {"type": "manifest_header", "emb": "embeddings/mm.pt"},
        _row("embeddings/nn.pt", g="agentic"),
    ]
    kept = disjoint_val_rows(train, val)
    assert [r["emb"] for r in kept] == [
        "embeddings/mm.pt",
        "embeddings/nn.pt",
    ]


def test_empty_train_keeps_all_val():
    val = [_row("embeddings/aa.pt")]
    assert disjoint_val_rows([], val) == val
