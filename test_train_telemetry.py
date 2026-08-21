"""Telemetry contracts for modal_train.py: TrainMonitor analytics + curve renderer.

These cover the pure, CPU-side logic so the GPU loop stays thin. Run:
    python -m pytest test_train_telemetry.py -q
"""
import json

from modal_train import TrainMonitor, render_curves


def _feed(m, n, loss=1.0, gnorm=0.5):
    for step in range(1, n + 1):
        m.update_train(step=step, loss=loss, grad_norm=gnorm,
                       lr=1e-4, tokens=128, step_ms=100.0)


def test_monitor_ema_converges_on_constant_series():
    m = TrainMonitor()
    _feed(m, 50)
    assert abs(m.loss_ema - 1.0) < 0.01
    assert abs(m.gnorm_ema - 0.5) < 0.01


def test_monitor_median_window():
    m = TrainMonitor(median_window=4)
    _feed(m, 4)
    assert m.loss_median() == 1.0
    assert m.gnorm_median() == 0.5


def test_no_spike_when_only_loss_spikes():
    alerts = []
    m = TrainMonitor(median_window=20, on_alert=alerts.append)
    _feed(m, 30)                       # stable baseline
    rec = m.update_train(step=31, loss=10.0, grad_norm=0.5,
                         lr=1e-4, tokens=128, step_ms=100.0)  # loss jumps, grad calm
    assert rec is None and alerts == []


def test_spike_requires_both_loss_and_grad_norm():
    alerts = []
    m = TrainMonitor(median_window=20, on_alert=alerts.append)
    _feed(m, 30)
    rec = m.update_train(step=31, loss=10.0, grad_norm=50.0,
                         lr=1e-4, tokens=128, step_ms=100.0)
    assert rec is not None and alerts == [rec]
    assert rec["type"] == "alert"
    assert rec["loss"] == 10.0 and rec["grad_norm"] == 50.0


def test_spike_cooldown_suppresses_repeat_alerts():
    alerts = []
    m = TrainMonitor(median_window=20, cooldown=100, on_alert=alerts.append)
    _feed(m, 30)
    m.update_train(step=31, loss=10.0, grad_norm=50.0, lr=1e-4, tokens=1, step_ms=1.0)
    m.update_train(step=32, loss=10.0, grad_norm=50.0, lr=1e-4, tokens=1, step_ms=1.0)
    assert len(alerts) == 1            # second spike inside cooldown is silent


def test_warmup_window_blocks_premature_alerts():
    # fewer than `min_history` samples: medians are meaningless -> never alert
    alerts = []
    m = TrainMonitor(median_window=200, min_history=10, on_alert=alerts.append)
    m.update_train(step=1, loss=99.0, grad_norm=999.0, lr=1e-4, tokens=1, step_ms=1.0)
    assert alerts == []


def test_render_curves_writes_png(tmp_path):
    recs = []
    for step in range(1, 101):
        recs.append({"type": "train", "step": step, "epoch": 0,
                     "loss": 2.0 / step, "grad_norm": 1.0 / step,
                     "lr": 5e-4, "tokens": 1024, "it_s": 2.0,
                     "samples_seen": step * 8, "peak_gib": 40.0, "ts": 0.0})
        if step % 25 == 0:
            recs.append({"type": "val", "step": step, "val_loss": 1.9 / step, "n_rows": 256})
    out = tmp_path / "train_curves.png"
    render_curves(recs, str(out), grok_lo=70, grok_hi=90)
    assert out.exists() and out.stat().st_size > 10_000
