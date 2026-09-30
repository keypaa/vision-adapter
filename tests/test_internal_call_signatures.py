"""No internal call may pass a kwarg the callee does not accept.

Regression 2026-09-30: a bulk edit threading a new parameter through the
run paths appended it to every `_streaming_train(...)` delegation AND to the
`geometry_guard(...)` call sitting between them. The suite stayed green —
nothing executed that line — and the first real GPU run died at startup with
"geometry_guard() got an unexpected keyword argument 'lr_horizon'".

A static check catches the whole class: it needs no execution, so it covers
the paths a unit test would never reach.

Run: python -m pytest tests/test_internal_call_signatures.py -q
"""
import ast
import inspect
from pathlib import Path

import vision_adapter.train as train
import vision_adapter.cli as cli

MODULES = [train, cli]


def _callee_signatures():
    sigs = {}
    for mod in MODULES:
        for name in dir(mod):
            obj = getattr(mod, name)
            if not callable(obj):
                continue
            try:
                sigs[name] = set(inspect.signature(obj).parameters)
            except (TypeError, ValueError):
                continue
    return sigs


def test_no_call_passes_an_unknown_kwarg():
    sigs = _callee_signatures()
    offenders = []
    for mod in MODULES:
        path = Path(inspect.getsourcefile(mod))
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id not in sigs:
                continue
            passed = {kw.arg for kw in node.keywords if kw.arg}
            passed |= {a.arg for a in node.args if hasattr(a, "arg")}
            unknown = passed - sigs[node.func.id]
            if unknown:
                offenders.append(
                    f"{path.name}:{node.lineno} {node.func.id}(...) unknown={sorted(unknown)}"
                )
    assert not offenders, "unknown kwargs:\n  " + "\n  ".join(offenders)


def test_the_specific_callee_is_callable_the_way_the_run_calls_it():
    """geometry_guard takes exactly (grid_source, allow_synthetic)."""
    params = set(inspect.signature(train.geometry_guard).parameters)
    assert params == {"grid_source", "allow_synthetic"}
    train.geometry_guard("measured", allow_synthetic=True)


def test_streaming_train_accepts_what_run_train_passes():
    """The run-to-stream delegation grew three kwargs; pin the set."""
    params = set(inspect.signature(train._streaming_train).parameters)
    assert {
        "resume_ckpt", "allow_synthetic", "lr_horizon",
    } <= params, params
