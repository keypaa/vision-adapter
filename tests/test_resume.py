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
        run_id="abc",
    )
    assert set(payload) == {"proj", "opt", "scaler", "step", "samples_seen",
                            "monitor", "rng", "plan", "cfg", "run_id"}
    assert payload["step"] == 100
    assert payload["samples_seen"] == 1600
    assert payload["run_id"] == "abc"


def test_resume_plan_pins_original_totals():
    from vision_adapter.train import _resolve_resume

    cfg = {"max_steps": 4000, "batch_size": 16, "seed": 0, "sample_size": 117600}
    r = _resolve_resume(ckpt_plan=cfg, cli_max_steps=8000, cli_seed=0)
    assert r["total_steps"] == 4000  # original wins, not the new CLI value
    assert r["start_pos_rows"] == r["resume_step"] * 16


def test_resume_reuses_run_id_and_appends(tmp_path):
    from vision_adapter.train import _open_resume_log

    log = tmp_path / "probe_log.jsonl"
    log.write_text('{"type": "config_header", "run_id": "abc"}\n{"type": "train", "step": 1}\n')
    fh, run_id = _open_resume_log(log, "abc")
    fh.write('{"type": "train", "step": 2}\n')
    fh.close()
    assert run_id == "abc"
    assert len(log.read_text().strip().splitlines()) == 3  # appended, not truncated
