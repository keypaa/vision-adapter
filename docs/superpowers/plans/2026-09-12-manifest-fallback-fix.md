# Manifest persistence + fallback removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A streaming failure mid-run never again dies with a misleading manifest `FileNotFoundError`, and a failed 4000-step run never reports success.

**Architecture:** Persist the HF-fetched manifest to `data_dir/train_manifest.jsonl` right after fetch (header-first, atomic); delete the automatic smoke fallback in `run_train` so streaming exceptions preserve artifacts and return exit code 1.

**Tech Stack:** Python, torch (untouched), pytest, ruff.

## Global Constraints

- Branch: `fix/resume-manifest-fallback` (already created, HEAD `454644e`).
- Commit only with `ruff ok` on touched files + `py_compile` + `pytest -q` green (pre-existing ruff errors in `vision_adapter/train.py`: F401 `time`, E702 ×2 line 39, F811 ×2 — do not touch, do not introduce new ones).
- No behavior change to the happy path (streaming train, dryrun, fake-fixture smoke).
- One task = one commit. Never mix Task 1 and Task 2 in a commit.

---

### Task 1: Persist fetched manifest to data_dir

**Files:**
- Modify: `vision_adapter/train.py` (~line 355-363, `_streaming_train` manifest fetch block)
- Modify: `vision_adapter/manifest.py` only if `write_manifest_with_header` signature needs checking (read first)
- Test: `tests/test_manifest_persist.py` (new file)

**Interfaces:**
- Consumes: `rows: list[dict]` from `_fetch_manifest(...)`, `write_manifest_with_header` from `vision_adapter.manifest` (verify exact signature by reading `manifest.py:89-120` before coding).
- Produces: `data_dir/train_manifest.jsonl` on disk (header-first, atomic write via tmp+replace), helper `_persist_fetched_manifest(data_dir: Path, rows: list[dict]) -> Path`.

- [ ] **Step 1: Read `write_manifest_with_header` signature**

Read: `vision_adapter/manifest.py:89-120`.
Expected: a function taking rows + path + git sha (or similar) and writing header-first atomically. Note exact parameter names.

- [ ] **Step 2: Write the failing test**

```python
"""Fetched manifest must land on disk so later stages never FileNotFoundError."""
from pathlib import Path

from vision_adapter.train import _persist_fetched_manifest
from vision_adapter.manifest import load_manifest


def test_persist_fetched_manifest_roundtrip(tmp_path):
    rows = [
        {"emb": "e1", "user": "u1", "assistant": "a1", "g": "t"},
        {"emb": "e2", "user": "u2", "assistant": "a2", "g": "t"},
    ]
    out = _persist_fetched_manifest(tmp_path, rows)
    assert out == tmp_path / "train_manifest.jsonl"
    back, header = load_manifest(out)
    assert len(back) == 2
    assert header is not None  # header-first, not legacy
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_manifest_persist.py -q`
Expected: FAIL with "cannot import name '_persist_fetched_manifest'" (ImportError).

- [ ] **Step 4: Write minimal implementation**

In `vision_adapter/train.py` (near `_fetch_manifest` usage), add:

```python
def _persist_fetched_manifest(data_dir: Path, rows: list[dict]) -> Path:
    """Write HF-fetched rows to data_dir/train_manifest.jsonl (header-first).

    Without this, rows live only in memory and every later stage that reads
    the local manifest (smoke fallback, local train, final push) crashes or
    silently skips. Atomic tmp+replace; never raises (logs and returns path).
    """
    from vision_adapter.manifest import write_manifest_with_header  # adjust names to actual signature

    out = Path(data_dir) / "train_manifest.jsonl"
    write_manifest_with_header(rows, out)  # adjust to actual signature
    return out
```

Then call it in `_streaming_train` right after `rows = _fetch_manifest(...)` (line ~363):

```python
    else:
        rows = _fetch_manifest(cache_dir=str(data_dir / "cache"), token=tok_hf)
        try:
            _persist_fetched_manifest(data_dir, rows)
        except Exception as e:  # noqa: BLE001 — persistence is best-effort, training must not die here
            print(f"[train] manifest persist failed ({e}) — continuing with in-memory rows", flush=True)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_manifest_persist.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add tests/test_manifest_persist.py vision_adapter/train.py
git commit -m "fix(train): persist HF-fetched manifest to data_dir"
```

---

### Task 2: Remove automatic smoke fallback on streaming failure

**Files:**
- Modify: `vision_adapter/train.py` (lines ~627-634, `run_train` except block)
- Test: extend `tests/test_manifest_persist.py` (same file, no new file)

**Interfaces:**
- Consumes: `vision_adapter.train._streaming_train`, `vision_adapter.train._smoke_train_with_fake_data` (both monkeypatched in test).
- Produces: `run_train` returns `1` on streaming exception; `_smoke_train_with_fake_data` is never invoked on that path.

- [ ] **Step 1: Write the failing test**

```python
def test_streaming_failure_returns_1_without_smoke(tmp_path, monkeypatch):
    import vision_adapter.train as tr
    from vision_adapter.config import probe_config

    monkeypatch.setattr(tr, "_streaming_train", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    def _no_smoke(*a, **k):
        raise AssertionError("smoke fallback must not run after a real streaming failure")
    monkeypatch.setattr(tr, "_smoke_train_with_fake_data", _no_smoke)

    rc = tr.run_train(tmp_path, probe_config(), backend=None, max_steps=5, device="cpu")
    assert rc == 1
```

Note: `device="cpu"` skips the GPU gate (`require_gpu`); missing manifest proceeds to `_streaming_train` (mocked). If `run_train` signature/behavior differs, read `run_train` first and adjust — the invariant is "returns nonzero, smoke untouched".

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest_persist.py::test_streaming_failure_returns_1_without_smoke -q`
Expected: FAIL (currently returns smoke result `0`, or the smoke function raises its own error).

- [ ] **Step 3: Write minimal implementation**

Replace (current lines ~627-634):

```python
    try:
        return _streaming_train(dd, cfg, max_steps, dev, dtype)
    except Exception as e:
        import traceback
        print(f"[train] streaming train failed ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        print("[train] falling back to smoke stub (check HF_TOKEN and network)", flush=True)
        return _smoke_train_with_fake_data(dd, cfg, max_steps, dev)
```

with:

```python
    try:
        return _streaming_train(dd, cfg, max_steps, dev, dtype)
    except Exception as e:
        import traceback
        print(f"[train] streaming train failed ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        print("[train] NOT falling back to smoke: partial ckpts/log preserved, exiting 1", flush=True)
        return 1
```

Keep the exact `_streaming_train(...)` argument list as-is (do not drop `dtype` if present — match the current call).

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_manifest_persist.py tests/test_cli.py -q`
Expected: all PASS.

- [ ] **Step 5: Full gate + commit**

Run: `pytest tests/ -q` (expect 79+2 passed), `ruff check` on touched files (no NEW errors vs the 5 pre-existing in train.py).
```bash
git add tests/test_manifest_persist.py vision_adapter/train.py
git commit -m "fix(train): no auto smoke fallback after streaming failure (exit 1)"
```

---

## Self-Review

- Spec coverage: A1 (persist) → Task 1; A4 (fallback) → Task 2. A2/A3 were findings, no code needed. B-items belong to `2026-09-12-training-resume.md`, not here.
- No placeholders: all code blocks concrete; Step 1 of Task 1 forces signature verification.
- Type consistency: `_persist_fetched_manifest(data_dir: Path, rows: list[dict]) -> Path` used identically in test and implementation.
