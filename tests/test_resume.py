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


def test_download_ckpt_picks_latest_or_exact(tmp_path, monkeypatch):
    import vision_adapter.train as tr

    files = ["projector_step100.pt", "projector_step200.pt", "probe_log.jsonl"]
    monkeypatch.setattr(tr, "_list_hf_ckpt_files", lambda repo: files)
    def _fake_dl(repo_id, filename, **kw):
        p = tmp_path / filename
        p.write_bytes(b"ckpt-bytes")
        return str(p)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_dl)

    got = tr._download_ckpt("keypa/x", None, tmp_path)
    assert got.name == "projector_step200.pt"  # latest when step None
    got = tr._download_ckpt("keypa/x", 100, tmp_path)
    assert got.name == "projector_step100.pt"


def test_resume_reuses_run_id_and_appends(tmp_path):
    from vision_adapter.train import _open_resume_log

    log = tmp_path / "probe_log.jsonl"
    log.write_text('{"type": "config_header", "run_id": "abc"}\n{"type": "train", "step": 1}\n')
    fh, run_id = _open_resume_log(log, "abc")
    fh.write('{"type": "train", "step": 2}\n')
    fh.close()
    assert run_id == "abc"
    assert len(log.read_text().strip().splitlines()) == 3  # appended, not truncated


def test_resume_matches_continuous_on_tiny_model(tmp_path):
    """5 continuous steps vs 3 + ckpt-restore-2 must give equal projector weights.

    The resume half goes through the real Task 1/2 restore path —
    build_ckpt_payload, torch.save/load round-trip, _resolve_resume extent
    pinning, and the same proj/opt/monitor/RNG restore sequence as
    _streaming_train — so this gates the production restore code, not a
    plain torch deepcopy.
    """
    import pytest
    import torch

    from tests.test_adaptive_ckpt import StubTok, _tiny_qwen
    from vision_adapter.core import (
        HourglassProjector,
        ProbeMonitor,
        make_collate,
        train_step_qwen,
    )
    from vision_adapter.train import (
        _collect_rng_state,
        _resolve_resume,
        _restore_rng_state,
        build_ckpt_payload,
    )

    tok = StubTok()
    hidden = 32
    total_steps = 5
    resume_at = 3
    batch_size = 2

    def _fresh():
        # Re-seed so every branch starts from IDENTICAL model/proj/opt state.
        # (A single top-level seed would hand each branch different random
        # inits and fail for the wrong reason.)
        torch.manual_seed(1234)
        m = _tiny_qwen(layers=2, vocab=256, hidden=hidden)
        p = HourglassProjector(4096, hidden)
        o = torch.optim.AdamW(p.parameters(), lr=1e-3)
        mon = ProbeMonitor()
        return m, p, o, mon

    def _batch(seed):
        g = torch.Generator().manual_seed(seed)
        items = [{"vis": torch.randn(4, 4096, generator=g), "user": "u",
                  "assistant": "answer", "g": "t"} for _ in range(batch_size)]
        return make_collate(tok, tok.pad_token_id, max_len=64)(items)

    def _step(model, proj, opt, mon, s):
        out = train_step_qwen(model, proj, opt, _batch(s), "cpu", adaptive_ckpt=None)
        assert out["finite"]
        mon.update(s + 1, out["loss"], (s + 1) * batch_size)
        return out

    # Path A: 5 continuous steps.
    model_a, proj_a, opt_a, mon_a = _fresh()
    for s in range(total_steps):
        _step(model_a, proj_a, opt_a, mon_a, s)

    # Path B: 3 steps, then checkpoint via the real Task 1 payload helper.
    model_b, proj_b, opt_b, mon_b = _fresh()
    for s in range(resume_at):
        _step(model_b, proj_b, opt_b, mon_b, s)
    payload = build_ckpt_payload(
        {k: v.cpu().clone() for k, v in proj_b.state_dict().items()},
        opt_b.state_dict(),
        None,
        step=resume_at,
        samples_seen=resume_at * batch_size,
        monitor_state=mon_b.to_dict(),
        rng_state=_collect_rng_state(),
        plan_meta={"manifest_sha256": "test", "seed": 0, "sample_size": 20,
                   "batch_size": batch_size, "stream_order_hash": "test",
                   "max_steps": total_steps},
        cfg_dict={"lr": 1e-3, "batch_size": batch_size},
        run_id="test-run",
    )
    ckpt_path = tmp_path / f"projector_step{resume_at}.pt"
    torch.save(payload, str(ckpt_path))

    # Path C: fresh objects, restored via the _streaming_train sequence.
    resume_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    merged_plan = dict(resume_ckpt.get("plan", {}) or {})
    merged_plan.setdefault("step", resume_ckpt.get("step", 0))
    info = _resolve_resume(merged_plan, cli_max_steps=5000)
    assert info["total_steps"] == total_steps  # original extent wins over CLI
    assert info["resume_step"] == resume_at
    assert info["start_pos_rows"] == resume_at * batch_size
    assert resume_ckpt.get("run_id") == "test-run"

    model_c, proj_c, opt_c, mon_c = _fresh()
    proj_c.load_state_dict(resume_ckpt["proj"])
    opt_c.load_state_dict(resume_ckpt["opt"])
    mon_c.load_state_dict(resume_ckpt.get("monitor", {}) or {})
    _restore_rng_state(resume_ckpt.get("rng"))
    for s in range(resume_at, total_steps):
        _step(model_c, proj_c, opt_c, mon_c, s)

    for pa, pc in zip(proj_a.parameters(), proj_c.parameters()):
        assert torch.allclose(pa, pc, atol=1e-6)
    assert mon_c.ema == pytest.approx(mon_a.ema)
