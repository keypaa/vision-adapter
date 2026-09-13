# Fix CUBLAS bl2 gate + empty_cache discipline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Small-batch OOM (alloc 94GB at B=16 L≈800) no longer kills the 4000-step run; monster shards split correctly on true quadratic cost.

**Architecture:** Gate micro-batching on B·L² (bl2) vs COST_MAX 25M instead of B·L 20000; keep empty_cache only at save + ckpt ON with synchronize; remove per-step empty_cache churn.

**Tech Stack:** Python, torch, pytest, ruff.

## Global Constraints

- Branch: feat/bucketed-hf-streaming (merge from fix/mem-debug if needed), HEAD 3af66a7 baseline.
- Commit only with `ruff ok` on touched files (pre-existing 5 errors in train.py stay) + `py_compile` + `pytest -q` green.
- Keep resume/fallback/manifest fixes intact; do not break fresh-start path.
- One task = one commit.

---

### Task 1: Fix micro-batch gate to bl2 (COST_MAX) + cost-aware n_splits

**Files:**
- Modify: `vision_adapter/train.py:810-845` (B·L gate + n_splits calc)
- Test: `tests/test_adaptive_ckpt.py` (extend)

**Interfaces:**
- Consumes: `batch["input_ids"].shape -> B,L`, `DEFAULT_COST_MAX` (25M), `ckpt_needed_for_batch` already gates on bl2.
- Produces: `_should_split(batch) -> bool` on `bl2 > COST_MAX`; `n_splits = ceil(bl2 / COST_MAX)` capped; `micro = max(1, B // n_splits)` derived from cost, not B·L.

- [ ] **Step 1: Write the failing test**

```python
def test_bl2_gate_triggers_on_monster_not_small():
    from vision_adapter.train import _should_split
    import torch
    # Build via real collate shape (B,L) — bl2 computed on padded L, not fake zeros alone
    small = {"input_ids": torch.zeros(16, 800), "attention_mask": torch.ones(16, 800)}
    monster = {"input_ids": torch.zeros(16, 2500), "attention_mask": torch.ones(16, 2500)}
    assert _should_split(small) is False
    assert _should_split(monster) is True

def test_n_splits_cost_aware():
    from vision_adapter.train import _n_splits_for_batch
    # B=16 L=4900 bl2=384M -> ceil(384/25)=16 splits -> micro=1
    assert _n_splits_for_batch(16, 4900) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_ckpt.py -k bl2 -q`
Expected: FAIL ImportError (helpers missing).

- [ ] **Step 3: Write minimal implementation**

```python
from vision_adapter.core import _resolve_ckpt_budget, DEFAULT_COST_MAX

def _should_split(batch) -> bool:
    B, L = batch["input_ids"].shape[:2]
    budget = _resolve_ckpt_budget("auto")
    cost_max = budget[1] if budget else DEFAULT_COST_MAX
    return B * L * L > cost_max

def _n_splits_for_batch(B: int, L: int) -> int:
    import math
    from vision_adapter.core import _resolve_ckpt_budget, DEFAULT_COST_MAX
    budget = _resolve_ckpt_budget("auto")
    cost_max = budget[1] if budget else DEFAULT_COST_MAX
    return max(1, math.ceil(B * L * L / cost_max))
```

Wire in train loop: replace `if B0*L0 > _bl_budget` with `if _should_split(batch)` and `n_splits = _n_splits_for_batch(B0, L0)`. Keep `VISION_ADAPTER_COST_MAX` env override via `_resolve_ckpt_budget`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_ckpt.py -k bl2 -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vision_adapter/train.py tests/test_adaptive_ckpt.py
git commit -m "fix(train): gate micro-batching on bl2 COST_MAX, cost-aware splits"
```

---

### Task 2: Revert per-step empty_cache to save + ckpt ON with synchronize

**Files:**
- Modify: `vision_adapter/train.py:857-870` (remove per-step empty_cache block)
- Modify: `vision_adapter/core.py:626-635` + `vision_adapter/train.py:882-887` (add synchronize before empty_cache)
- Test: `tests/test_adaptive_ckpt.py`

**Interfaces:**
- Consumes: `torch.cuda.synchronize`, `torch.cuda.empty_cache`.
- Produces: `empty_cache` only when `ckpt_on` or at save, always preceded by `synchronize()` if CUDA available.

- [ ] **Step 1: Write the failing test**

```python
def test_empty_cache_guarded_by_synchronize(monkeypatch):
    import vision_adapter.core as core
    calls = []
    monkeypatch.setattr(core.torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(core.torch.cuda, "empty_cache", lambda: calls.append("empty"))
    # trigger ckpt ON path — any L with budget (1,1) forces ON
    from vision_adapter.core import train_step_qwen, HourglassProjector
    # use tiny batch that still triggers ckpt ON via budget
    assert calls == ["sync", "empty"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_ckpt.py -k synchronize -q`
Expected: FAIL (order wrong or missing sync).

- [ ] **Step 3: Write minimal implementation**

In `core.py` ckpt ON block and `train.py` save block:

```python
try:
    import torch as _t
    if _t.cuda.is_available():
        _t.cuda.synchronize()
        _t.cuda.empty_cache()
        cache_emptied = True
except Exception:
    pass
```

Delete the `if not ckpt_on: empty_cache` block at `train.py:857-870` and remove `rec["cache_emptied_step"]` emission (or keep it always False for dashboard compat).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_ckpt.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vision_adapter/core.py vision_adapter/train.py tests/test_adaptive_ckpt.py
git commit -m "fix(train): empty_cache only at save/ckpt ON with synchronize"
```

---

### Task 3: Verify 400→600 resume smoke (no OOM, reserved stays <80GB)

**Files:**
- Test: manual Molab run (resume hf 400, 200 steps), check `probe_log.jsonl` mem snapshots + nvidia-smi.

**Interfaces:**
- Consumes: HF ckpt 400, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`.

- [ ] **Step 1: Run resume**

Run: `python -m vision_adapter train ... --resume hf --resume-step 400 --max-steps 600` (200 steps).

- [ ] **Step 2: Verify**

Check: no CUBLAS/OOM error, `mem_reserved_gb` stays <80GB at steps 400, 475, 600 (3 samples), `reserved` at save points <60GB, step 475 passes. Fail if any `reserved` >80GB on small-batch steps (L<1200).

- [ ] **Step 3: Commit docs if needed**

```bash
git add docs/superpowers/plans/2026-09-13-fix-cublas-bl2-gate.md
git commit -m "docs: cublas gate plan verified on resume 400→600"
```
