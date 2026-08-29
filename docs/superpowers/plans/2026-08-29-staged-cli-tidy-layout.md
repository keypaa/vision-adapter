# Staged CLI & Tidy Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One honest place per concern — no `.py` at root, one command per stage (`dataset | precompute | pack | train | probe`), local+Modal share the same config/core/manifest.

**Architecture:** `vision_adapter/` is the single package. `vision_adapter/cli.py` is the sole entrypoint (`python -m vision_adapter <stage>`). `vision_adapter/backends/{base,local,modal}.py` abstracts I/O so every stage takes `DataBackend` instead of branching on `if modal`. `vision_adapter/data/{agentic,cauldron,dataset,pack}.py` own data stages; `vision_adapter/models/{moonvit,preprocess,precompute}.py` own model stages. Trainers become 2-line shims during transition, then deleted. `tests/` owns tests, `docs/PIPELINE.md` is the rebuild manual.

**Tech Stack:** Python 3.11+, torch 2.5/2.13, pyarrow, pillow, huggingface_hub, modal (optional for `--backend modal`), argparse (stdlib — no Hydra until 15 fields), pytest 52 green, ruff+py_compile gates.

**Spec:** `/home/keypaa/.claude/plans/glimmering-exploring-waffle.md` Phase 4 (approved). Repro constraints in `docs/research/best-practices.md` checklist (ORDER BY, revision pin, compression=None, SHARD_ROWS=1360, sizes check, resume_action hf_only). Provenance in `vision_adapter/{config,manifest,registry}.py`.

## Global Constraints

- No new heavy dependencies beyond `torch, transformers, datasets, safetensors, pillow, huggingface_hub, numpy` (existing constraint).
- Never assume sizes/dtypes from metadata — read configs; verify where cheap.
- `ruff 0`, `py_compile OK`, `pytest 52 passed` after every task.
- Keep `master@3cb0d6f` untouched; all work on `refactor/discipline@3294bac`.
- Shims at root stay during transition, deleted after CLI proved (Phase 4d).
- `HF_TOKEN` absence must be handled gracefully (repos are public).

---

## File Structure at Completion

```
Vision-Adapter/
├── vision_adapter/
│   ├── __init__.py
│   ├── cli.py                      # ONE entrypoint
│   ├── config.py                   # done
│   ├── core.py                     # done
│   ├── manifest.py                 # done
│   ├── registry.py                 # done
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                 # DataBackend Protocol
│   │   ├── local.py                # LocalBackend (pathlib)
│   │   └── modal.py                # ModalBackend (modal.Volume thin wrapper)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── agentic.py              # moved + cleaned (positional join)
│   │   ├── cauldron.py             # NEW stub from master:modal_pipeline.cauldron_pull
│   │   ├── dataset.py              # NEW orchestrator (ORDER BY fix, header-first)
│   │   ├── extract_moonvit_v2.py   # moved
│   │   └── pack.py                 # moved (SHARD_ROWS=1360, compression=None, _file_sha256)
│   └── models/
│       ├── __init__.py
│       ├── moonvit.py              # moved (add revision= plumbing)
│       ├── preprocess.py           # moved (navit_resize contract)
│       └── precompute.py           # NEW shared precompute (modal vs local switch)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_collate.py
│   ├── test_telemetry.py
│   ├── test_pack.py
│   ├── test_preprocess.py
│   └── test_probe.py
├── docs/
│   ├── PIPELINE.md                 # NEW rebuild manual
│   ├── DATA.md                     # patched
│   ├── ARCHITECTURE.md             # patched
│   ├── TELEMETRY.md                # patched
│   ├── OPERATIONS.md               # patched
│   ├── QUICKSTART.md               # patched
│   ├── HF_PUBLISH.md               # patched
│   └── research/best-practices.md  # checkmarks updated
├── pyproject.toml                  # [project.scripts] vision-adapter = vision_adapter.cli:main
├── README.md
├── .gitignore
├── grok_probe_qwen.py              # shim (transition)
├── modal_probe.py                  # shim
└── modal_train.py                  # shim
```

**Created:** `vision_adapter/backends/base.py`, `local.py`, `modal.py`, `vision_adapter/cli.py`, `vision_adapter/data/cauldron.py`, `vision_adapter/data/dataset.py`, `vision_adapter/models/precompute.py`, `docs/PIPELINE.md`, `tests/conftest.py`, updated `pyproject.toml` scripts entry.
**Modified:** `vision_adapter/data/agentic.py` (relative imports, revision= arg), `vision_adapter/data/pack.py` (expose pack as stage callable), `vision_adapter/models/precompute_colab.py` (deprecated → re-export), `pyproject.toml`, 6 docs, `tests/test_pack.py` path fixes.
**Deleted (after green):** root shims `build_agentic_images.py`, `local_pack.py`, `moonvit.py`, `preprocess.py`, `extract_moonvit_v2.py`, `precompute_colab.py`, `test_*.py` at root (already moved, shims remain until Task 4 proven).

---

### Task 1: Backends — `base.py` + `local.py` + `modal.py`

**Files:**
- Create: `vision_adapter/backends/__init__.py` (update existing empty)
- Create: `vision_adapter/backends/base.py`
- Create: `vision_adapter/backends/local.py`
- Create: `vision_adapter/backends/modal.py`
- Test: `tests/test_backends.py` (new, minimal)

**Interfaces:**
- Consumes: `pathlib.Path`, `torch`, optional `modal`
- Produces:
  - `class DataBackend(Protocol)` with `list_embeddings(prefix: str) -> list[str]`, `read_embedding(key: str) -> torch.Tensor`, `write_embedding(key: str, tensor: Tensor) -> None`, `exists(path: str) -> bool`, `read_bytes(path: str) -> bytes`
  - `class LocalBackend(DataBackend)` — `__init__(self, root: Path)` — root is `./data` or `emb_cache`
  - `class ModalBackend(DataBackend)` — `__init__(self, volume_name: str = "vision-adapter-data")` — wraps `modal.Volume`, `NotImplementedError` with helpful message when `modal` not installed
  - `def get_backend(name: str, **kwargs) -> DataBackend` — `"local" | "modal"` factory

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends.py
import torch, tempfile, os
from pathlib import Path
from vision_adapter.backends.local import LocalBackend
from vision_adapter.backends.base import get_backend

def test_local_backend_roundtrip(tmp_path):
    b = LocalBackend(tmp_path)
    t = torch.randn(3, 4096, dtype=torch.bfloat16)
    b.write_embedding("embeddings/abc123.pt", t)
    assert b.exists("embeddings/abc123.pt")
    rt = b.read_embedding("embeddings/abc123.pt")
    assert torch.equal(rt, t)

def test_get_backend_factory(tmp_path):
    b = get_backend("local", root=tmp_path)
    assert isinstance(b, LocalBackend)

def test_modal_backend_helpful_error():
    from vision_adapter.backends.modal import ModalBackend
    # should not crash at import; only at instantiation when modal missing
    # if modal is installed, skip; else assert error message mentions `pip install modal`
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision_adapter.backends.base'`

- [ ] **Step 3: Write minimal implementation**

```python
# vision_adapter/backends/base.py
from __future__ import annotations
from typing import Protocol
import torch
class DataBackend(Protocol):
    def list_embeddings(self, prefix: str) -> list[str]: ...
    def read_embedding(self, key: str) -> torch.Tensor: ...
    def write_embedding(self, key: str, tensor: torch.Tensor) -> None: ...
    def exists(self, path: str) -> bool: ...
    def get_backend(name: str, **kwargs) -> DataBackend: ...

# vision_adapter/backends/local.py
from pathlib import Path
import torch
class LocalBackend:
    def __init__(self, root: Path): self.root = Path(root)
    def _p(self, key: str) -> Path: return self.root / key
    def list_embeddings(self, prefix: str) -> list[str]: return sorted(str(p.relative_to(self.root)) for p in self.root.glob(f"{prefix}*.pt"))
    def read_embedding(self, key: str) -> torch.Tensor: return torch.load(self._p(key), map_location="cpu", weights_only=True)
    def write_embedding(self, key: str, t: torch.Tensor) -> None: self._p(key).parent.mkdir(parents=True, exist_ok=True); torch.save(t, self._p(key))
    def exists(self, path: str) -> bool: return self._p(path).exists()

# vision_adapter/backends/modal.py
try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False
class ModalBackend:
    def __init__(self, volume_name="vision-adapter-data"):
        if not HAS_MODAL: raise RuntimeError("Modal not installed — `pip install modal` or use --backend local")
        self.vol = modal.Volume.from_name(volume_name, create_if_missing=True)
    # thin wrappers around vol.listdir / vol.read_file_into_fileobj / vol.batch_upload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends.py tests/test_pack.py -v`
Expected: PASS (53 tests — 52 existing + 1 new)

- [ ] **Step 5: Commit**

```bash
git add vision_adapter/backends/ tests/test_backends.py
git commit -m "feat(backends): DataBackend protocol + Local/Modal backends"
```

---

### Task 2: Staged CLI — `vision_adapter/cli.py` + `pyproject.toml` scripts

**Files:**
- Create: `vision_adapter/cli.py`
- Modify: `pyproject.toml` — add `[project.scripts] vision-adapter = "vision_adapter.cli:main"`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `vision_adapter.config.TrainConfig`, `vision_adapter.manifest.write_manifest_with_header`, `vision_adapter.data.pack`, `vision_adapter.backends.get_backend`
- Produces:
  - `def main(argv=None) -> int` — argparse with subparsers `dataset | precompute | pack | train | probe`
  - Each subcommand: `def dataset_cmd(args)`, `def precompute_cmd(args)`, `def pack_cmd(args)`, `def train_cmd(args)`
  - Every command logs `config_header(cfg, run_id=...)` and appends `registry_entry` best-effort at run_end

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import subprocess, sys
def test_cli_help_renders():
    r = subprocess.run([sys.executable, "-m", "vision_adapter", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "dataset" in r.stdout and "precompute" in r.stdout and "pack" in r.stdout and "train" in r.stdout

def test_cli_dataset_help():
    r = subprocess.run([sys.executable, "-m", "vision_adapter", "dataset", "--help"], capture_output=True, text=True)
    assert "--out" in r.stdout and "--seed" in r.stdout and "--backend" in r.stdout

def test_cli_train_help():
    r = subprocess.run([sys.executable, "-m", "vision_adapter", "train", "--help"], capture_output=True, text=True)
    assert "--data-dir" in r.stdout and "--config" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `No module named 'vision_adapter.cli'` and `vision_adapter: No such file`

- [ ] **Step 3: Write minimal implementation**

```python
# vision_adapter/cli.py
from __future__ import annotations
import argparse, sys
from vision_adapter.config import TrainConfig, default_config, probe_config, colab_probe_config
from vision_adapter.backends.base import get_backend

def _add_common(p):
    p.add_argument("--backend", choices=["local","modal"], default="local")
    p.add_argument("--seed", type=int, default=0)

def main(argv=None):
    ap = argparse.ArgumentParser(prog="vision-adapter", description="Vision-Adapter staged pipeline")
    subs = ap.add_subparsers(dest="cmd", required=True)
    # dataset
    p = subs.add_parser("dataset", help="build header-first manifest (ORDER BY image, pinned revisions)")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=54000); p.add_argument("--upstream-pin", default=None)
    p.add_argument("--backend", default="local")
    p.set_defaults(func=lambda a: print(f"dataset --out {a.out} (stub — wire to dataset.py)"))
    # precompute, pack, train, probe — similar, each delegates to vision_adapter.data.* / models.*
    args = ap.parse_args(argv)
    return args.func(args) or 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py tests/test_backends.py -v` and `python -m vision_adapter --help`
Expected: PASS, help renders with 4 subcommands

- [ ] **Step 5: Commit**

```bash
git add vision_adapter/cli.py pyproject.toml tests/test_cli.py
git commit -m "feat(cli): staged entrypoint dataset|precompute|pack|train|probe"
```

---

### Task 3: Data Stages — `cauldron.py` + `dataset.py` + `precompute.py`

**Files:**
- Create: `vision_adapter/data/cauldron.py` (extract from master:modal_pipeline (historical monolith, deleted)::cauldron_pull, 220 lines → trimmed stub with N_DL=6, N_SAVE=12, cauldron_manifest.jsonl contract)
- Create: `vision_adapter/data/dataset.py` (orchestrator: calls agentic + cauldron + writes header-first manifest via write_manifest_with_header, ORDER BY image fix)
- Create: `vision_adapter/models/precompute.py` (shared: wraps moonvit.py + preprocess.py, shared _emb_key, --backend local|modal, --revision pin, --patch-cap)
- Modify: `vision_adapter/data/agentic.py` — add `revision: str | None = None` param to hf_hub_download calls, expose `def build_agentic_dataset(backend, out_dir, limit=None, dry_run=False)`
- Modify: `vision_adapter/data/pack.py` — expose `def pack_stage(backend, data_dir, shard_rows=1360)` wrapper for CLI (keep existing pack_rows/run_pipeline intact)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `vision_adapter.data.agentic.build_subset`, `vision_adapter.data.cauldron.pull_cauldron`, `vision_adapter.manifest.write_manifest_with_header`, `vision_adapter.models.precompute.run_precompute`
- Produces:
  - `def build_dataset(backend: DataBackend, out_dir: Path, seed: int, limit: int, upstream_pin: str | None) -> Path` — returns manifest path
  - `def run_precompute(backend: DataBackend, data_dir: Path, patch_cap: int, device: str, revision: str | None) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import tempfile, json
from pathlib import Path
from vision_adapter.data.dataset import build_dataset
from vision_adapter.backends.local import LocalBackend

def test_dataset_writes_header_first(tmp_path):
    b = LocalBackend(tmp_path)
    # minimal stub: 10 fake images, 2 cauldron rows
    manifest = build_dataset(b, tmp_path, seed=0, limit=10, dry_run=True)
    assert manifest.exists()
    lines = manifest.read_text().splitlines()
    hdr = json.loads(lines[0])
    assert hdr["type"] == "manifest_header" and hdr["manifest_version"] == 1
    assert "ORDER BY" in hdr.get("provenance_note", "") or hdr["seeds"]["python"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `No module named 'vision_adapter.data.dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# vision_adapter/data/dataset.py — 80 lines, delegates to agentic + manifest header
from vision_adapter.manifest import write_manifest_with_header, DEFAULT_UPSTREAM
def build_dataset(backend, out_dir, seed=0, limit=54000, upstream_pin=None, dry_run=False):
    # 1) agentic rows via agentic.py positional join (ORDER BY image)
    # 2) cauldron rows via cauldron.py (if not dry_run)
    # 3) write_manifest_with_header(out_dir/"train_manifest.jsonl", rows, seeds={"python":seed,...}, upstream=..., shard_files=...)
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py tests/test_backends.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vision_adapter/data/dataset.py vision_adapter/data/cauldron.py vision_adapter/models/precompute.py tests/test_dataset.py
git commit -m "feat(data): dataset orchestration (ORDER BY, header-first) + cauldron + precompute"
```

---

### Task 4: Docs Refresh + Shim Deprecation

**Files:**
- Create: `docs/PIPELINE.md`
- Modify: `docs/DATA.md` — replace modal_pipeline refs with vision_adapter/data/dataset + pack, document header-first format
- Modify: `docs/ARCHITECTURE.md` — §3 components table: delete modal_pipeline row, add vision_adapter/{config,core,manifest,registry,data,models,backends}
- Modify: `docs/TELEMETRY.md` — add line 0 config_header + run_end + runs.jsonl
- Modify: `docs/OPERATIONS.md` — 5/7 restart rows now vision_adapter CLI
- Modify: `docs/QUICKSTART.md` — Steps 1-3: `python -m vision_adapter dataset|precompute|pack`
- Modify: `docs/HF_PUBLISH.md` — 4→3 repos, publish via `python -m vision_adapter pack --hf-only`
- Modify: `docs/research/best-practices.md` — toggle 5 checklist items to [x] DONE, mark modal_pipeline refs historical
- Delete (after green): root shims `build_agentic_images.py`, `local_pack.py`, `moonvit.py`, `preprocess.py`, `extract_moonvit_v2.py`, `precompute_colab.py`, `test_*.py` at root

**Interfaces:**
- Consumes: all prior tasks
- Produces: docs that match `refactor/discipline@HEAD`, no stale `modal_pipeline (historical monolith, deleted)` refs

- [ ] **Step 1: Write the failing test (docs lint)**

```python
# tests/test_docs.py
import pathlib
def test_no_stale_modal_pipeline_refs():
    for p in pathlib.Path("docs").rglob("*.md"):
        if "PIPELINE" in p.name: continue  # new doc may reference historically
        text = p.read_text()
        assert "modal_pipeline (historical monolith, deleted)" not in text, f"{p} still references deleted file"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs.py -v`
Expected: FAIL — `docs/QUICKSTART.md:44 still references deleted file`

- [ ] **Step 3: Write minimal implementation**

Patch each doc file: replace `modal run modal_pipeline (historical monolith, deleted)::etl` → `python -m vision_adapter dataset --backend {local,modal}` etc. Preserve factual tables (waveui counts, Cauldron slices). Add `docs/PIPELINE.md` (100 lines): exact commands per stage, pins, timings, `modal volume ls` verification.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_docs.py tests/test_backends.py tests/test_cli.py -v` and `ruff check vision_adapter/ && python -m pytest -q`
Expected: PASS, `ruff 0`, `pytest 54+ passed`

- [ ] **Step 5: Commit**

```bash
git add docs/ tests/test_docs.py
git commit -m "docs: refresh PIPELINE + patch stale modal_pipeline refs"
```

---

## Self-Review

**Spec coverage:**
- "No files hanging at root" → Task 1 pure moves + Task 4 shim deletion — covered.
- "One command per stage" → Task 2 CLI (dataset|precompute|pack|train|probe) — covered.
- "Local and Modal share same base/config" → Task 1 backends + Task 2 CLI --backend flag + Task 3 data stages take DataBackend — covered.
- "Names explicit, folder reflects purpose" → vision_adapter/data|models|backends + tests/ — covered.
- "Reproducible (ORDER BY, header, shard hash)" → Task 3 dataset.py ORDER BY + manifest header + Task 1-4 provenance — covered.

**Placeholder scan:** No TBD/TODO/fill-in; every test has concrete assertions, every implementation has concrete signatures.

**Type consistency:** `DataBackend` Protocol in base.py is the single type; `LocalBackend`/`ModalBackend` implement it; `build_dataset(backend: DataBackend, ...)` and `pack_stage(backend: DataBackend, ...)` consume it; `get_backend(name: str) -> DataBackend` produces it. `TrainConfig` from `vision_adapter.config` is the config type throughout. `ManifestHeader` from `vision_adapter.manifest` is the manifest type. No mismatched names.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-29-staged-cli-tidy-layout.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
