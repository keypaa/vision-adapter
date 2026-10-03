"""gen_check.py defaults to the STORED embedding, not noise.

Measured 2026-10-03: with a noise visual span the action verb is unpredictable,
so the model falls back to its narrative prior. The same checkpoint emits
`click(start_box=[500,300])` on real embeddings. That single default produced a
wrong conclusion for a full run, so it is pinned here.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen_check.py"


def _source() -> str:
    return SCRIPT.read_text()


def test_noise_is_opt_in_not_default():
    """--no-real-emb must exist and be off unless asked for."""
    src = _source()
    assert '"--no-real-emb"' in src, "gen_check lost its noise escape hatch"
    ap = next(a for a in ast.walk(ast.parse(src))
              if isinstance(a, ast.Call) and getattr(a.func, "attr", "") == "add_argument"
              and "--no-real-emb" in ast.dump(a))
    flags = [kw.value.value for kw in ap.keywords
             if kw.arg == "action" and isinstance(kw.value, ast.Constant)]
    assert flags == ["store_true"], "--no-real-emb must be a flag (store_true)"


def test_generation_prefers_real_embeddings_over_randn():
    """The generate loop must not call torch.randn unconditionally."""
    src = _source()
    loop = src[src.index("outputs = []"):]
    loop = loop[:loop.index("def _report_scores")]
    assert "torch.randn" in loop, "noise fallback should still exist"
    # ...but it has to sit behind the guard
    assert "if real is not None:" in loop, "real embeddings are not preferred"
    noise_line = next(ln for ln in loop.splitlines() if "torch.randn" in ln)
    guard = loop.index("if real is not None:")
    assert loop.index(noise_line) > guard, \
        "torch.randn runs before the real-embedding guard"
    assert "[WARN] noise visual span" in loop, \
        "a noise run must say so — a silent fallback reads as a real result"


def test_scores_reported_so_a_centre_default_reads_as_failure():
    """A `[500,300]` constant must not pass as grounding."""
    src = _source()
    assert "def _report_scores" in src
    assert "verb correct" in src and "syntax correct" in src
    assert "within 100 px" in src, "no coordinate score means [500,300] looks fine"
    assert src.index("def _report_scores") < src.rindex("raise SystemExit"), \
        "the scores are computed but never called"


def test_reuses_the_proven_shard_reader():
    """Don't reimplement shard streaming; visual_ablation's is tested."""
    src = _source()
    assert "from visual_ablation import _load_real_embeddings" in src


def test_unreachable_embeddings_degrade_to_noise_with_a_warning(capsys):
    """An unreachable shard must warn and fall back, not crash — and the
    warning must say the verb can't be judged."""
    import importlib.util
    import sys

    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("gen_check_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # force the reader to fail the way an offline machine would
    import visual_ablation
    original = visual_ablation._load_real_embeddings
    visual_ablation._load_real_embeddings = lambda *a, **k: (_ for _ in ()).throw(
        OSError("no network"))
    try:
        out = mod._real_embeddings(None, [{"emb": "x"}])
    finally:
        visual_ablation._load_real_embeddings = original

    assert out is None, "a failed shard read must return None, not raise"
    printed = capsys.readouterr().out
    assert "falling back to noise" in printed, "the degradation must be announced"


def test_script_still_parses():
    r = subprocess.run([sys.executable, "-c", f"import ast;ast.parse(open({str(SCRIPT)!r}).read())"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr