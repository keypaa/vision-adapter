"""Step ckpts must carry full resumable state, not just weights."""
from vision_adapter.train import build_ckpt_payload


def test_ckpt_payload_has_all_resume_keys():
    payload = build_ckpt_payload(
        proj_state={"w": 1}, opt_state={"s": 2}, scaler_state=None,
        step=100, samples_seen=1600, monitor_state={"ema": 1.5},
        rng_state={"python": [1], "torch": [2]},
        plan_meta={"manifest_sha256": "abc", "seed": 0, "sample_size": 3200,
                   "batch_size": 16, "stream_order_hash": "def"},
        cfg_dict={"lr": 5e-4, "batch_size": 16},
    )
    assert set(payload) == {"proj", "opt", "scaler", "step", "samples_seen",
                            "monitor", "rng", "plan", "cfg"}
    assert payload["step"] == 100
    assert payload["samples_seen"] == 1600
