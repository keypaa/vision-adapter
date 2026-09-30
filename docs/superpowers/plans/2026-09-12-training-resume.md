# Training resume (local + HF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A killed 4000-step run resumes from the last pushed/saved ckpt (weights + optimizer + schedule + data position) from local disk or HF, and continues the identical trajectory.

**Architecture:** Extend step ckpts to full training state; add `--resume {off,local,hf} --resume-step K`; on resume, restore weights/optimizer/scaler/monitor/RNG, pin LR total-steps and the epoch plan from ckpt metadata, skip `K*batch_size` rows via existing `start_pos`, append to the same `probe_log.jsonl`/`run_id`.

**Tech Stack:** Python, torch, huggingface_hub (download side), pytest, ruff.

## Global Constraints

- Branch: `fix/resume-manifest-fallback` (same branch as manifest plan; stack resume commits AFTER manifest commits).
- Commit only with `ruff ok` on touched files (no new errors) + `py_compile` + `pytest -q` green.
- Full determinism (bit-identical trajectory) is NOT required; statistical equivalence is (same loss curve shape, no loss spike >2× at resume step, monitor EMA continuous). Document this in code.
- Never break the fresh-start path: default `--resume off` behaves exactly as today.
- One task = one commit.

---

### Task 1: Full-state step checkpoints

**Files:**
- Modify: `vision_adapter/train.py` (step-save block ~line 508-520, final-save ~line 531+)
- Test: `tests/test_resume.py` (new file)

**Interfaces:**
- Consumes: `proj`, `opt`, `scaler`, `monitor`, `cfg`, plan meta (manifest hash, seed, sample_size, batch_size). Read exact construction sites first: projector (~line 400), opt (~403), scaler (~404), monitor (~406), `lr_at` in `core.py`, `ProbeMonitor` fields in `core.py:216-239`.
- Produces: `build_ckpt_payload(...) -> dict` and step files containing keys `{proj, opt, scaler, step, samples_seen, monitor, rng, plan, cfg}`. Later tasks consume these exact key names — do not rename.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resume.py -q`
Expected: FAIL with "cannot import name 'build_ckpt_payload'" (ImportError).

- [ ] **Step 3: Write minimal implementation**

```python
CKPT_KEYS = ("proj", "opt", "scaler", "step", "samples_seen",
             "monitor", "rng", "plan", "cfg")


def _collect_rng_state() -> dict:
    import random

    import torch

    state: dict = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def build_ckpt_payload(proj_state, opt_state, scaler_state, step: int,
                       samples_seen: int, monitor_state: dict,
                       rng_state: dict, plan_meta: dict, cfg_dict: dict) -> dict:
    return {"proj": proj_state, "opt": opt_state, "scaler": scaler_state,
            "step": step, "samples_seen": samples_seen,
            "monitor": monitor_state, "rng": rng_state,
            "plan": plan_meta, "cfg": cfg_dict}
```

Wire into the step-save block: replace
`_torch.save({"proj": proj.state_dict(), "step": step, "loss": rec["loss"]}, ...)`
keeping the filename `projector_step{step}.pt` and the push call. Gather:
- `opt_state = opt.state_dict()`, `scaler_state = scaler.state_dict() if scaler else None`
- `monitor_state = monitor.to_dict()` — if `ProbeMonitor` has no `to_dict`, add a minimal one in `core.py` returning `{ema, prev_ema, history, ema_history, last_banner, last_alert_step, n_alerts, n_banners, collapse_step}` plus matching `load_state_dict` (small, same task, same commit).
- `plan_meta`: manifest sha (reuse `manifest_sha256`), seed `0`, current `sample_size`, `cfg.batch_size`, sha1 of `stream_order` list.
- `cfg_dict = cfg.to_dict()`.

Keep `loss` out or in — your choice, but keep the `rec["loss"]` print untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resume.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_resume.py vision_adapter/train.py vision_adapter/core.py
git commit -m "feat(train): full-state step ckpts (opt/scaler/monitor/rng/plan/cfg)"
```

---

### Task 2: `--resume local` (same machine)

**Files:**
- Modify: `vision_adapter/cli.py` (train parser ~line 254-275: add `--resume {off,local,hf}` default `off`, `--resume-step int|None` default None meaning latest)
- Modify: `vision_adapter/train.py` (`run_train` signature + `_streaming_train` restore path)
- Test: extend `tests/test_resume.py`

**Interfaces:**
- Consumes: `build_ckpt_payload` keys from Task 1; `EmbStreamDataset(..., start_pos=...)` (already exists in `stream.py`); `lr_at(step, total_steps, ...)` in `core.py`.
- Produces: `run_train(..., resume="local", resume_step=K)` continues at step K+1 with restored opt/monitor, LR computed against ORIGINAL total steps, log appended under the SAME run_id.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_resume.py -q`
Expected: FAIL with ImportError on `_resolve_resume`.

- [ ] **Step 3: Write minimal implementation**

```python
def _resolve_resume(ckpt_plan: dict, cli_max_steps: int | None, cli_seed: int = 0) -> dict:
    """Pin training extent from ckpt metadata, never from new CLI values."""
    total = int(ckpt_plan["max_steps"]) if "max_steps" in ckpt_plan else None
    if total is None:  # backward compat: old ckpts without max_steps
        total = int(cli_max_steps or 0)
    return {"total_steps": total, "seed": int(ckpt_plan.get("seed", cli_seed)),
            "sample_size": int(ckpt_plan.get("sample_size", 0)),
            "batch_size": int(ckpt_plan.get("batch_size", 16)),
            "resume_step": int(ckpt_plan.get("step", 0)),
            "start_pos_rows": int(ckpt_plan.get("step", 0)) * int(ckpt_plan.get("batch_size", 16))}


def _open_resume_log(log_path, run_id):
    """Append to the existing run log, preserving run_id."""
    fh = open(log_path, "a", buffering=1)
    return fh, run_id
```

Restore path in `_streaming_train` (new params `resume_ckpt: dict | None = None`):
- load `proj.load_state_dict(ckpt["proj"])`, `opt.load_state_dict(ckpt["opt"])`, scaler if present, monitor restore, RNG restore (best-effort per key).
- `start_step = ckpt["step"] + 1`; loop `for step in range(start_step, total+1)` where `total` comes from `_resolve_resume` (NOT the new `max_steps`).
- pass `start_pos=start_pos_rows` to the FIRST `EmbDS(...)` only (epoch-wrap `ds2` keeps 0 — document why in a comment).
- open log with `_open_resume_log` reusing ckpt run_id (store run_id in ckpt payload Task 1 — if missing, add `"run_id"` to `build_ckpt_payload` now and update Task 1's test key set + implementation in the SAME commit as this task; do not amend the Task 1 commit).

CLI: add `--resume choices(off,local,hf) default off`, `--resume-step type=int default=None`; `train_cmd` passes them into `run_train(...)`; `run_train` loads latest (or K) `projector_step*.pt` from `_cache_root`/`data_dir` when `resume == "local"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_resume.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_resume.py vision_adapter/train.py vision_adapter/cli.py vision_adapter/core.py
git commit -m "feat(train): --resume local with opt/schedule/data restore"
```

---

### Task 3: `--resume hf` (fresh session)

**Files:**
- Modify: `vision_adapter/train.py` (download helper + wire into resume path)
- Test: extend `tests/test_resume.py`

**Interfaces:**
- Consumes: `_push_file_to_hf` repo contract (flat names `projector_step{K}.pt`); `hf_hub_download` (monkeypatched in test).
- Produces: `_download_ckpt(repo_id: str, step: int | None, dest_dir: Path) -> Path` returning the local file; Task 2's restore path reuses it unchanged.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resume.py::test_download_ckpt_picks_latest_or_exact -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
def _list_hf_ckpt_files(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo_id, repo_type="model")


def _download_ckpt(repo_id: str, step: int | None, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    names = [f for f in _list_hf_ckpt_files(repo_id)
             if f.startswith("projector_step") and f.endswith(".pt")]
    if not names:
        raise FileNotFoundError(f"no step ckpts in hf:{repo_id}")
    if step is None:
        def _n(nm: str) -> int:
            try:
                return int(nm.removeprefix("projector_step").removesuffix(".pt"))
            except ValueError:
                return -1
        filename = max(names, key=_n)
    else:
        filename = f"projector_step{step}.pt"
        if filename not in names:
            raise FileNotFoundError(f"{filename} not in hf:{repo_id}")
    local = hf_hub_download(repo_id, filename, repo_type="model",
                            local_dir=str(dest_dir))
    return Path(local)
```

Wire: when `resume == "hf"`, call `_download_ckpt(repo, resume_step, data_dir / "cache")` then run the identical Task 2 restore path (refactor the restore into `_restore_from_ckpt_dict(ckpt, ...)` if not already — same task, same commit).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_resume.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_resume.py vision_adapter/train.py
git commit -m "feat(train): --resume hf downloads step ckpt then restores"
```

---

### Task 4: Resume equivalence gate (CPU) + docs

**Files:**
- Modify: `tests/test_resume.py` (equivalence test on tiny CPU model)
- Modify: `docs/NEXT_STEPS.md` (mark resume done, document flags)
- Test: same file

**Interfaces:**
- Consumes: `train_step_qwen` with `adaptive_ckpt=None` (deterministic CPU path), `build_ckpt_payload`, Task 2 restore helpers.
- Produces: proof that interrupted-then-resumed == continuous (same seed, tiny model, few steps).

- [ ] **Step 1: Write the failing test**

```python
def test_resume_matches_continuous_on_tiny_model():
    """5 continuous steps vs 3 + resume-2 must give equal projector weights."""
    import copy

    import torch

    from vision_adapter.core import HourglassProjector, make_collate, train_step_qwen
    from tests.test_adaptive_ckpt import StubTok, _tiny_qwen  # reuse hermetic fixtures

    torch.manual_seed(0)
    tok = StubTok()
    hidden = 32

    def _fresh():
        m = _tiny_qwen(layers=2, vocab=256, hidden=hidden)
        p = HourglassProjector(4096, hidden)
        o = torch.optim.AdamW(p.parameters(), lr=1e-3)
        return m, p, o

    def _batch(seed):
        g = torch.Generator().manual_seed(seed)
        items = [{"vis": torch.randn(4, 4096, generator=g), "user": "u",
                  "assistant": "answer", "g": "t"} for _ in range(2)]
        return make_collate(tok, tok.pad_token_id, max_len=64)(items)

    model_a, proj_a, opt_a = _fresh()
    for s in range(5):
        train_step_qwen(model_a, proj_a, opt_a, _batch(s), "cpu", adaptive_ckpt=None)

    model_b, proj_b, opt_b = _fresh()
    for s in range(3):
        train_step_qwen(model_b, proj_b, opt_b, _batch(s), "cpu", adaptive_ckpt=None)
    # simulate resume: carry optimizer state + weights into fresh objects
    saved_opt = copy.deepcopy(opt_b.state_dict())
    model_c, proj_c, opt_c = _fresh()
    proj_c.load_state_dict(copy.deepcopy(proj_b.state_dict()))
    opt_c.load_state_dict(saved_opt)
    for s in range(3, 5):
        train_step_qwen(model_c, proj_c, opt_c, _batch(s), "cpu", adaptive_ckpt=None)

    for pa, pc in zip(proj_a.parameters(), proj_c.parameters()):
        assert torch.allclose(pa, pc, atol=1e-6)
```

Note: if `tests/test_adaptive_ckpt.py` fixtures are not importable as shown (leading underscore + module path), inline the tiny-model + stub builder in this file instead — same shapes, no new dependency. Check importability in Step 2; on failure, inline and re-run (still RED until restore path exists — this test passes with plain torch ops, so gate it: it must FAIL until Task 2's `_resolve_resume` is used... if it passes immediately, strengthen it by routing the resume half through `_restore_from_ckpt_dict` from Task 3's refactor).

- [ ] **Step 2: Run test, confirm meaningful signal**

Run: `pytest tests/test_resume.py::test_resume_matches_continuous_on_tiny_model -q`
Expected: PASS only if restore path is real (if it passes trivially, route through Task 2/3 helpers per the note).

- [ ] **Step 3: Update docs/NEXT_STEPS.md**

Mark resume items done; document `--resume off|local|hf`, `--resume-step`, required env, and the statistical-equivalence (not bit-identical) guarantee.

- [ ] **Step 4: Full gate + commit**

Run: `pytest tests/ -q` (all green), `ruff check` touched files (no new errors).
```bash
git add tests/test_resume.py docs/NEXT_STEPS.md vision_adapter/train.py
git commit -m "test(train): resume equivalence gate + docs"
```

---

## Self-Review

- Spec coverage: B1 → Task 1; B2/B3-core → Task 2; B4 → Task 3; equivalence proof → Task 4. Manifest items live in the sibling plan, not here.
- No placeholders: every step has exact code/commands; Task 2/Step 3 names the fallback if fixtures don't import.
- Type consistency: ckpt keys fixed in Task 1 (`proj, opt, scaler, step, samples_seen, monitor, rng, plan, cfg` + `run_id` added in Task 2) and reused verbatim in Tasks 2–4; `_resolve_resume`/`_open_resume_log`/`_download_ckpt` signatures frozen at definition.
