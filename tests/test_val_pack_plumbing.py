"""build_val_plan picks the pack when it exists, and refuses a leaky one.

The fallback is measured at ~10 min per probe (77 rows over 21 shards), so the
pack path has to actually engage — and when it does, the disjointness has to be
re-verified, because a pack rebuilt by hand after the train plan changed could
otherwise serve rows the trainer has seen.
"""
from __future__ import annotations

import json

import pytest

from vision_adapter.train import build_val_plan


def _write_pack(tmp_path, source_keys, source_to_emb, index):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vision_adapter.data.pack import SCHEMA

    pack = tmp_path / "val_pack.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"key": k, "n_vis": 374, "vis_bytes": b"\x00" * 8} for k in source_keys],
            schema=SCHEMA,
        ),
        pack,
    )
    pack.with_suffix(".map.json").write_text(json.dumps(source_to_emb))
    return pack


def _vals(embs):
    return [{"emb": e, "user": "u", "assistant": "a", "g": "agentic",
             "grid_thw": [1, 27, 27]} for e in embs]


def test_pack_is_used_when_present(tmp_path):
    # the packed rows must live on shards the trainer is NOT streaming, or the
    # pack is correctly refused (see the leak test below)
    index = {"a.pt": ("data/emb_0002.parquet", 0, 374),
             "b.pt": ("data/emb_0002.parquet", 1, 374)}
    _write_pack(tmp_path, ["src_a", "src_b"], {"src_a": "a.pt", "src_b": "b.pt"}, index)

    order, plan, n, local = build_val_plan(
        val_rows=_vals(["a.pt", "b.pt"]), index=index,
        stream_order=["data/emb_0050.parquet"],
        train_shards={"data/emb_0050.parquet"},
        data_dir=tmp_path, excluded=set(), seed=0,
        build_plan=lambda *a, **k: pytest.fail("must not build a multi-shard plan"),
    )

    assert n == 2
    assert order == ["val_pack.parquet"]
    assert local == {"val_pack.parquet": str(tmp_path / "val_pack.parquet")}
    assert [r["_row"] for r in plan["val_pack.parquet"]] == [0, 1], \
        "_row must be the position in the pack, since the reader indexes by it"


def test_a_leaky_pack_falls_back_and_says_why(tmp_path, capsys):
    """A packed row that resolves to a train shard must not be served."""
    index = {"a.pt": ("data/emb_0002.parquet", 0, 374)}
    _write_pack(tmp_path, ["src_a"], {"src_a": "a.pt"}, index)

    order, plan, n, local = build_val_plan(
        val_rows=_vals(["a.pt"]), index=index,
        stream_order=["data/emb_0002.parquet"],
        train_shards={"data/emb_0002.parquet"},   # <- the leak
        data_dir=tmp_path, excluded=set(), seed=0,
        build_plan=lambda rows, idx, **k: {"data/emb_0002.parquet": rows[:1]},
    )

    assert local == {}, "a rejected pack must not be wired in"
    assert "val pack unusable" in capsys.readouterr().out


def test_a_pack_whose_map_does_not_match_its_keys_is_refused(tmp_path):
    """Half-written pack: the map is from an earlier run."""
    index = {"a.pt": ("data/emb_0002.parquet", 0, 374),
             "b.pt": ("data/emb_0003.parquet", 1, 374)}
    _write_pack(tmp_path, ["src_a", "src_b"], {"src_a": "a.pt"}, index)  # missing b

    _, _, _, local = build_val_plan(
        val_rows=_vals(["a.pt", "b.pt"]), index=index,
        stream_order=["data/emb_0003.parquet"],
        train_shards=set(), data_dir=tmp_path, excluded=set(), seed=0,
        build_plan=lambda rows, idx, **k: {},
    )
    assert local == {}, "a mismatched pack must be refused, not half-used"


def test_a_missing_sidecar_refuses_the_pack(tmp_path):
    index = {"a.pt": ("data/emb_0002.parquet", 0, 374)}
    pack = tmp_path / "val_pack.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vision_adapter.data.pack import SCHEMA
    pq.write_table(pa.Table.from_pylist(
        [{"key": "src_a", "n_vis": 374, "vis_bytes": b"\x00" * 8}], schema=SCHEMA), pack)

    _, _, _, local = build_val_plan(
        val_rows=_vals(["a.pt"]), index=index, stream_order=[],
        train_shards=set(), data_dir=tmp_path, excluded=set(), seed=0,
        build_plan=lambda rows, idx, **k: {},
    )
    assert local == {}, "without the map there is no way to join pack to manifest"


def test_without_a_pack_the_fallback_still_excludes_train_shards(tmp_path):
    """The old behaviour must survive when no pack exists."""
    index = {"a.pt": ("data/emb_0002.parquet", 0, 374)}
    seen = {}

    def fake_plan(rows, idx, sample_size, seed, excluded_shards):
        seen.update(sample_size=sample_size, seed=seed, excluded=excluded_shards)
        return {"data/emb_0003.parquet": rows}

    order, plan, n, local = build_val_plan(
        val_rows=_vals(["a.pt"]), index=index,
        stream_order=["data/emb_0002.parquet", "data/emb_0003.parquet"],
        train_shards={"data/emb_0002.parquet"},
        data_dir=tmp_path, excluded={"data/emb_0000.parquet"}, seed=7,
        build_plan=fake_plan,
    )

    assert seen["excluded"] == {"data/emb_0002.parquet"}
    assert seen["seed"] == 7
    # `excluded` is the hardcoded emb_0000/0001 holdout, not the train shards
    assert order == ["data/emb_0002.parquet", "data/emb_0003.parquet"]
    assert local == {}


def test_a_pack_missing_from_the_val_manifest_is_refused(tmp_path):
    """Keys that are in the pack but not in the manifest have no text to pair
    with — a 10 % gap is tolerated, a 50 % one is a stale manifest."""
    index = {f"k{i}.pt": ("data/emb_0090.parquet", i, 374) for i in range(10)}
    src = {f"src_{i}": f"k{i}.pt" for i in range(10)}
    _write_pack(tmp_path, list(src), src, index)

    _, _, _, local = build_val_plan(
        val_rows=_vals([f"k{i}.pt" for i in range(3)]),   # only 3 of 10
        index=index, stream_order=[], train_shards=set(),
        data_dir=tmp_path, excluded=set(), seed=0,
        build_plan=lambda rows, idx, **k: {},
    )
    assert local == {}, "a mostly-unmatched pack must be refused"


def test_local_shards_skip_the_download(monkeypatch, tmp_path):
    """EmbStreamDataset must read a local parquet without asking HF."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    import vision_adapter.data.stream as st
    from vision_adapter.data.pack import SCHEMA

    asked = []

    def _boom(name, cache_dir=None):
        asked.append(name)
        return None

    monkeypatch.setattr(st, "_get_hf_shard_path", _boom)
    monkeypatch.setattr(st, "_download_shard_hf_transfer", _boom)

    payload = torch.zeros(2, 4096, dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    local = tmp_path / "val_pack.parquet"
    pq.write_table(pa.Table.from_pylist(
        [{"key": "k", "n_vis": 2, "vis_bytes": payload}], schema=SCHEMA), local)

    rows = [{"emb": "e", "user": "u", "assistant": "a", "g": "g",
             "grid_thw": [1, 27, 27], "_row": 0}]
    ds = st.EmbStreamDataset({"val_pack.parquet": rows}, ["val_pack.parquet"],
                             local_shards={"val_pack.parquet": str(local)})
    got = list(ds.__iter__())

    assert len(got) == 1, "the local pack must actually be read"
    assert got[0]["vis"].shape == (2, 4096)
    assert asked == [], "nothing may be fetched from HF for a local shard"