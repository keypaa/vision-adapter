"""The val probe must not re-stream 17GB of shards on every probe.

Measured 2026-09-30 on the T4 run: the val plan spans 99 shards, and a full
pass Range-fetches ~17GB (~170 MiB per shard). Re-streaming that per probe
is untenable over a long run — but the val itself is only ~10MB of vectors
(1240 rows x 4096 dims x bf16), i.e. 0.91% of the 136k-row corpus.

So the probe materializes the val once, into the volume's cache, and every
later probe reads it locally. Cost: one pass, ever.

Run: python -m pytest tests/test_val_cache.py -q
"""
import torch

from vision_adapter.train import _val_cache_path, materialize_val


def _batch(n, dim=8, tag="t"):
    return {
        "input_ids": torch.arange(n, dtype=torch.long).view(1, -1),
        "attention_mask": torch.ones(1, n, dtype=torch.long),
        "labels": torch.arange(n, dtype=torch.long).view(1, -1),
        "vis": torch.randn(n, dim),
        "n_vis": torch.tensor([n - 2]),
        "g": [tag],
    }


def test_cache_path_is_under_the_run_data_dir(tmp_path):
    p = _val_cache_path(tmp_path, sample_size=10)
    assert p.parent == tmp_path
    assert p.suffix == ".pt"
    # a different val size must not reuse another val's cache
    assert p != _val_cache_path(tmp_path, sample_size=20)


def test_second_probe_reads_the_cache_without_streaming(tmp_path):
    """The regression: a materialized val must make the second pass free."""
    calls = []

    def batches():
        calls.append(1)
        return [_batch(6), _batch(4)]

    path = _val_cache_path(tmp_path, 2)
    materialize_val(batches(), tmp_path, sample_size=2)
    assert path.is_file()

    # Second call: a source that explodes if it is ever iterated. The cache
    # guard must short-circuit before touching it.
    class _Exploding:
        def __iter__(self):
            raise AssertionError("re-streamed the val on a cached probe")

    cached = materialize_val(_Exploding(), tmp_path, sample_size=2)
    assert len(cached) == 2
    assert len(calls) == 1          # the source ran exactly once
    assert cached[0]["vis"].shape[0] == 6


def test_cache_roundtrip_preserves_the_collate_contract(tmp_path):
    """Everything the loss needs must survive the roundtrip."""
    src = [_batch(8, dim=16)]
    path = materialize_val(src, tmp_path, sample_size=1)
    (got,) = materialize_val(None, tmp_path, sample_size=1)
    for k in ("input_ids", "attention_mask", "labels", "n_vis"):
        assert torch.equal(got[k], src[0][k]), k
    assert torch.equal(got["vis"], src[0]["vis"])
    assert got["g"] == src[0]["g"]


def test_no_source_and_no_cache_raises_rather_than_silently_probing_nothing(tmp_path):
    import pytest

    with pytest.raises(RuntimeError, match="val cache"):
        materialize_val(None, tmp_path / "empty", sample_size=1)
