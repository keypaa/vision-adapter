"""The held-out probe's slice filter must not break the packed val path.

The val split is 85% agentic but also carries doc/conv rows. When the run
trains on one slice, the probe is narrowed to it — but the narrowing has to
happen INSIDE build_val_plan, after the pack's coverage and leak checks:

if the manifest rows were narrowed first, the other groups' packs would look
broken ("only 0/126 keys are in the val manifest"), the whole pack path would
raise, and the run would fall back to the multi-shard plan — ~260 GiB per
probe, ~6 h of extra I/O for a 4,000-step run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_adapter.train import build_val_plan


def _write_pack(d: Path, group: str, embs):
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys = [f"{group}-k{i}" for i in range(len(embs))]
    pq.write_table(pa.table({"key": keys}), str(d / f"val_pack_{group}.parquet"))
    (d / f"val_pack_{group}.map.json").write_text(
        json.dumps(dict(zip(keys, embs))))


def test_the_pack_path_survives_the_slice_filter(tmp_path):
    ag = [f"embeddings/a{i}.pt" for i in range(3)]
    doc = [f"embeddings/d{i}.pt" for i in range(2)]
    _write_pack(tmp_path, "agentic", ag)
    _write_pack(tmp_path, "doc", doc)

    rows = [{"emb": e, "g": "agentic"} for e in ag] + \
           [{"emb": e, "g": "doc"} for e in doc]
    index = {e: ("data/emb_0009.parquet", 0, 364) for e in ag + doc}

    order, plan, n_total, local = build_val_plan(
        val_rows=rows, index=index, stream_order=["data/emb_0009.parquet"],
        train_shards=set(), data_dir=tmp_path, excluded=set(), seed=0,
        build_plan=lambda *a, **k: {}, groups="agentic",
    )

    assert n_total == 3, "only the agentic rows are probed"
    assert len(plan) == 1, "the doc pack is dropped, NOT a fallback"
    assert list(plan.values())[0][0]["g"] == "agentic"
    assert len(local) == 1


def test_without_a_filter_every_pack_is_used(tmp_path):
    ag = [f"embeddings/a{i}.pt" for i in range(3)]
    doc = [f"embeddings/d{i}.pt" for i in range(2)]
    _write_pack(tmp_path, "agentic", ag)
    _write_pack(tmp_path, "doc", doc)
    rows = [{"emb": e, "g": "agentic"} for e in ag] + \
           [{"emb": e, "g": "doc"} for e in doc]
    index = {e: ("data/emb_0009.parquet", 0, 364) for e in ag + doc}

    _order, plan, n_total, _local = build_val_plan(
        val_rows=rows, index=index, stream_order=["data/emb_0009.parquet"],
        train_shards=set(), data_dir=tmp_path, excluded=set(), seed=0,
        build_plan=lambda *a, **k: {}, groups="",
    )
    assert n_total == 5
    assert len(plan) == 2
