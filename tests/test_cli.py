import subprocess
import sys


def test_cli_help_renders():
    r = subprocess.run(
        [sys.executable, "-m", "vision_adapter", "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "dataset" in r.stdout and "precompute" in r.stdout and "pack" in r.stdout and "train" in r.stdout


def test_cli_dataset_help():
    r = subprocess.run(
        [sys.executable, "-m", "vision_adapter", "dataset", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--out" in r.stdout and "--seed" in r.stdout and "--backend" in r.stdout


def test_cli_train_help():
    r = subprocess.run(
        [sys.executable, "-m", "vision_adapter", "train", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--data-dir" in r.stdout and "--config" in r.stdout
