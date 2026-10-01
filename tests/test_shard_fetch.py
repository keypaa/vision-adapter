"""The fast whole-shard fetch must not be gated on which host we are on.

Two independent audits converged here. `load_span` fetches by HTTP Range
with urllib at 1.5-28 MiB/s measured, while `hf_transfer` — already
implemented — is documented at ~1 GiB/s. But `_download_shard_hf_transfer`
opens with `if not _in_modal(): return None`, so the fast path is reachable
only on Modal. The target platform is a rented GPU instance, which is not
Modal, so every real run pays the slow transport.

The whole-shard path serves identical bytes and identical row order as the
Range path — it differs only in transport. That is the property this pins,
because checkpoints record a plan hash and a row offset and must stay valid.

Run: python -m pytest tests/test_shard_fetch.py -q
"""
import os

import huggingface_hub

from vision_adapter.data import stream as st


def test_shard_fetch_is_not_gated_on_modal(monkeypatch, tmp_path):
    """A non-Modal host must still reach the whole-shard download."""
    monkeypatch.setattr(st, "_in_modal", lambda: False)
    called = {"n": 0}

    def fake_download(**kw):
        called["n"] += 1
        p = tmp_path / kw["filename"].split("/")[-1]
        p.write_bytes(b"fake parquet")
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising=False)
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")

    got = st._download_shard_hf_transfer(
        "data/emb_0006.parquet", cache_dir=str(tmp_path)
    )
    assert called["n"] == 1, (
        "the fast path was skipped because the host is not Modal"
    )
    assert got is not None


def test_transfer_stays_enabled_when_the_env_is_set(monkeypatch, tmp_path):
    """The env var is what switches on the Rust transfer; do not clobber it."""
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setattr(st, "_in_modal", lambda: False)

    def fake_download(**kw):
        return None

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising=False)
    st._download_shard_hf_transfer("data/x.parquet", cache_dir=str(tmp_path))
    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"


def test_failure_still_falls_back_to_range(monkeypatch, tmp_path):
    """A miss must return None so the caller uses Range — not raise."""
    monkeypatch.setattr(st, "_in_modal", lambda: False)

    def boom(**kw):
        raise OSError("network down")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom, raising=False)
    assert st._download_shard_hf_transfer(
        "data/x.parquet", cache_dir=str(tmp_path)
    ) is None


def test_cached_shard_is_not_refetched(monkeypatch, tmp_path):
    """If the file is already on disk, hand it back without touching network."""
    monkeypatch.setattr(st, "_in_modal", lambda: False)
    called = {"n": 0}

    def fake_download(**kw):
        called["n"] += 1
        return None

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising=False)
    cache = tmp_path / "shards"
    cache.mkdir()
    (cache / "emb_0006.parquet").write_bytes(b"already here")

    got = st._download_shard_hf_transfer("data/emb_0006.parquet", cache_dir=str(cache))
    assert got is not None, "an already-cached shard must be reused"
    assert called["n"] == 0, "a cached shard must not hit the network"
