"""Two protections for the run's raw evidence.

1. The log is pushed to the same HF repo as the checkpoints, under its plain
   name -- so an untagged run overwrites the PREVIOUS run's log in the shared
   repo. The tag scopes the name, so both survive.

2. A log file can hold several runs (the trainer appends on resume; a reused
   box keeps the old file). The analysis must read the LAST run, not the
   first, or the numbers describe a mixture of two experiments.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_probe import load_log
from vision_adapter.train import _log_filename


def test_the_log_name_is_scoped_by_the_run_tag(monkeypatch):
    monkeypatch.delenv("VISION_ADAPTER_CKPT_TAG", raising=False)
    assert _log_filename("probe_log.jsonl") == "probe_log.jsonl"
    assert _log_filename("probe_curves.png") == "probe_curves.png"
    monkeypatch.setenv("VISION_ADAPTER_CKPT_TAG", "p05")
    assert _log_filename("probe_log.jsonl") == "probe_log_p05.jsonl"
    assert _log_filename("probe_curves.png") == "probe_curves_p05.png"


def test_the_analysis_reads_the_last_run_in_an_appended_log(tmp_path):
    p = tmp_path / "probe_log.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"type": "config_header", "run_id": "old"}) + "\n")
        f.write(json.dumps({"type": "train", "step": 1, "loss": 1.0}) + "\n")
        f.write(json.dumps({"type": "run_end", "wall_min": 10}) + "\n")
        f.write(json.dumps({"type": "config_header", "run_id": "new"}) + "\n")
        f.write(json.dumps({"type": "train", "step": 1, "loss": 0.5}) + "\n")
        f.write(json.dumps({"type": "train", "step": 2, "loss": 0.4}) + "\n")
        f.write(json.dumps({"type": "run_end", "wall_min": 20}) + "\n")
    hdr, tr, end = load_log(p)
    assert hdr["run_id"] == "new"
    assert len(tr) == 2 and tr[0]["loss"] == 0.5
    assert end["wall_min"] == 20
