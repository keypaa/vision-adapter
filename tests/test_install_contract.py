"""A fresh install must provide everything training imports.

Audit 2026-10-01: `pip install -e .` installed only `torch, pyarrow,
pillow, accelerate, sentencepiece, hf_transfer` — and not transformers, so a
fresh machine could not run `vision-adapter train` at all. numpy and
safetensors were imported by the code and unlisted too. The optional
`train` extra listed transformers but nothing told anyone to install it.

These pins the contract: every third-party module imported at module scope by
the shipped package is declared, and the exact versions a known-good run used
are recorded in requirements-lock.txt.

Run: python -m pytest tests/test_install_contract.py -q
"""
import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "requirements-lock.txt"

# import name -> distribution name, where they differ
IMPORT_TO_DIST = {"PIL": "pillow"}
NOT_A_DEP = {
    "torch", "numpy", "PIL", "pyarrow", "transformers", "safetensors",
    "huggingface_hub", "accelerate", "sentencepiece", "datasets",
    "hf_transfer",  # optional extra
}


def _declared() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    out = set()
    for spec in data["project"]["dependencies"]:
        out.add(spec.split(">=")[0].split("==")[0].split("[")[0].strip())
    return out


def _third_party_imports() -> set[str]:
    """Top-level imports across the shipped package."""
    found = set()
    for py in sorted((ROOT / "vision_adapter").rglob("*.py")):
        tree = ast.parse(py.read_text())
        for node in tree.body:          # module scope only, not lazy imports
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found & NOT_A_DEP


def test_every_module_scope_import_is_declared():
    declared = _declared()
    imports = {IMPORT_TO_DIST.get(m, m) for m in _third_party_imports()}
    missing = imports - declared
    assert not missing, (
        f"imported at module scope but not in [project].dependencies: {sorted(missing)}"
    )


def test_training_needs_transformers():
    """The CLI's train path loads AutoModelForCausalLM — a fresh install must
    have it without asking for an extra nobody remembers."""
    assert "transformers" in _declared()


def test_lock_file_pins_every_runtime_dep():
    pinned = {
        line.split("==")[0].strip()
        for line in LOCK.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {"torch", "transformers", "numpy", "pyarrow", "huggingface_hub"} <= pinned


def test_lock_file_has_no_unpinned_requirement():
    """Every line must be an exact `==` pin; a bare name defeats the point."""
    for line in LOCK.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        assert "==" in s, f"{s!r} is not pinned — use name==version"


@pytest.mark.parametrize("f", ["requirements-lock.txt", "requirements-optional.txt"])
def test_requirement_files_exist(f):
    assert (ROOT / f).is_file(), f"{f} is missing — the install is not reproducible"
