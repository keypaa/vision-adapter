"""val_every must be controllable and fail loudly when misconfigured.

Audit 2026-09-30: the streaming path had no validation at all, then gained
one gated on cfg.val_every=250. But the interval could only be changed by
editing Python — no CLI flag — and nothing rejected val_every<=0, which
silently disabled the very probe this work added. A run would then look
healthy while being unmonitored.

Run: python -m pytest tests/test_val_every_control.py -q
"""
import subprocess
import sys

import pytest

from vision_adapter.config import TrainConfig, default_config


def test_val_every_rejects_zero_and_negative():
    """The failure mode this guards: a silently unmonitored run."""
    for bad in (0, -1, -250):
        with pytest.raises(ValueError, match="val_every"):
            TrainConfig(val_every=bad)


def test_val_every_still_constructs_and_reports_itself():
    cfg = default_config(val_every=50)
    assert cfg.val_every == 50
    assert cfg.to_dict()["val_every"] == 50


def test_cli_exposes_val_every():
    """A flag that does not exist cannot be controlled from a shell run."""
    out = subprocess.run(
        [sys.executable, "-m", "vision_adapter.cli", "train", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-400:]
    assert "--val-every" in out.stdout


def test_cli_val_every_default_matches_the_config_default():
    """The flag's documented default must equal the config's, not drift from it."""
    out = subprocess.run(
        [sys.executable, "-m", "vision_adapter.cli", "train", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    lines = out.stdout.splitlines()
    # skip the usage line, which also mentions --val-every
    idx = next(
        i for i, ln in enumerate(lines)
        if ln.strip().startswith("--val-every")
    )
    block = " ".join(lines[idx:idx + 3])
    assert str(default_config().val_every) in block, block
