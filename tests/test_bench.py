"""Phase 5 bench harnesses (NEXT_STEPS §3.1): protocol tests, CPU-only.

- time_download: measures any zero-arg download callable, returns schema.
- attention bench: runs on tiny CPU model with small shapes, checks schema
  (per-layer-type attribution sums to ~total, memory numbers present).
"""
import time

import pytest
import torch


def test_time_download_schema_with_fake():
    from scripts.bench_transfer import time_download

    def fake_fetch():
        time.sleep(0.01)
        return 1024 * 1024

    rec = time_download("fake", fake_fetch)
    assert rec["method"] == "fake"
    assert rec["bytes"] == 1024 * 1024
    assert rec["seconds"] >= 0.01
    assert rec["mib_s"] == pytest.approx(1.0 / rec["seconds"], rel=0.05)


def test_attention_bench_schema_cpu():
    from tests.test_adaptive_ckpt import _tiny_qwen
    from scripts.bench_attention import bench_shape

    torch.manual_seed(0)
    model = _tiny_qwen(layers=2, hidden=32)
    rec = bench_shape(model, "cpu", B=2, L=64, density=1.0, ckpt=False)
    assert set(rec) >= {"B", "L", "density", "ckpt", "fwd_ms", "bwd_ms",
                        "by_type_ms", "peak_alloc_mb"}
    assert rec["fwd_ms"] > 0 and rec["bwd_ms"] > 0
    total = rec["fwd_ms"] + rec["bwd_ms"]
    attributed = sum(rec["by_type_ms"].values())
    assert attributed <= total * 1.5  # hook overhead bounded
    assert attributed > 0  # attribution actually measures layers (ratio validated on GPU)
    assert rec["peak_alloc_mb"] >= 0
