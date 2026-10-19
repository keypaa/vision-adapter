#!/usr/bin/env python3
"""Analyse a probe run: where did the wall clock go, and WHICH SLICE LEARNS.

Reads a probe_log.jsonl (the trainer's per-step log) and answers the three
questions the 4,000-step probe left open (docs/DATA_PATH_REBUILD.md):

  1. the wall-clock split: GPU (step_ms) vs data wait (wait_ms) vs the rest
  2. is there a GROK CLIFF (the sharp loss drop the source recipe reports at
     ~step 900)?
  3. which slice is learning? The loss is a token-mean over the batch, so the
     per-step group mix (g_mix) is what lets a slice be compared fairly.

No GPU, no network, no replay of the data stream -- everything comes from the
log itself.

Usage:
    python scripts/analyze_probe.py --log data/probe_log.jsonl
    python scripts/analyze_probe.py --log data/probe_log.jsonl --out report.md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def load_log(path: Path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    hdr = next((r for r in recs if r.get("type") == "config_header"), None)
    tr = [r for r in recs if r.get("type") == "train"]
    end = next((r for r in recs if r.get("type") == "run_end"), None)
    return hdr, tr, end


def wall_clock_section(tr, end, p) -> None:
    p("\n## wall clock")
    gpu = sum(r.get("step_ms", 0.0) for r in tr) / 1000.0
    wait = sum(r.get("wait_ms", 0.0) for r in tr) / 1000.0
    ts = [r.get("ts") for r in tr if r.get("ts")]
    wall = (max(ts) - min(ts)) if len(ts) > 1 else None
    if wall:
        p(f"  wall (ts span)      {wall:,.0f} s")
        p(f"  GPU (sum step_ms)   {gpu:,.0f} s   ({100*gpu/wall:.1f}%)")
        p(f"  data wait (wait_ms) {wait:,.0f} s   ({100*wait/wall:.1f}%)")
        p(f"  other (log/ckpt)   {wall-gpu-wait:,.0f} s   ({100*(wall-gpu-wait)/wall:.1f}%)")
    else:
        p(f"  GPU (sum step_ms)   {gpu:,.0f} s   (no ts in this log)")
    if end:
        p(f"  run_end wall_min    {end.get('wall_min')} min   final_loss={end.get('final_loss')}")
    if wait == 0.0 and tr:
        p("  (this log predates wait_ms: the gap is wall - step_ms, which mixes "
          "network and local time)")


def loss_section(tr, window: int, p) -> list[tuple[int, float, float]]:
    p(f"\n## loss (per {window}-step window)")
    p(f"  {'window':>14} {'loss':>8} {'ema':>8} {'gnorm':>9} {'tokens':>8} {'L':>6}")
    curve = []
    for s0 in range(tr[0]["step"], tr[-1]["step"] + 1, window):
        w = [r for r in tr if s0 < r["step"] <= s0 + window]
        if not w:
            continue
        ema = w[-1].get("ema_loss")
        p(f"  {s0+1:>6}-{s0+window:<7} {st.mean(r['loss'] for r in w):>8.3f} "
          f"{'n/a' if ema is None else f'{ema:.3f}':>8} "
          f"{st.mean(r.get('gnorm', 0) for r in w):>9.2f} "
          f"{st.mean(r.get('tokens', 0) for r in w):>8.0f} "
          f"{st.mean(r.get('L', 0) for r in w):>6.0f}")
        curve.append((s0 + window, st.mean(r["loss"] for r in w), ema))
    emas = [(c[0], c[2]) for c in curve if c[2] is not None]
    if len(emas) > 1:
        drops = [(emas[i][0], emas[i - 1][1] - emas[i][1]) for i in range(1, len(emas))
                 if emas[i - 1][1] is not None and emas[i][1] is not None]
        if drops:
            # The recipe's grok is a sharp drop AFTER the plateau. The first
            # ~800 steps are the ordinary fast learning phase (the ema swings
            # there), so a big move there is not a cliff -- report separately.
            early = [d for d in drops if d[0] <= 800]
            late = [d for d in drops if d[0] > 800]
            if early:
                w = max(early, key=lambda d: d[1])
                p(f"  initial learning (<= step 800): largest ema move {w[1]:+.3f} at step ~{w[0]}"
                  f"  (not the grok)")
            if late:
                w = max(late, key=lambda d: d[1])
                vals = [v for _, v in emas if _ > 800]
                p(f"  GROK WINDOW (after step 800): ema ranges {min(vals):.3f}..{max(vals):.3f} "
                  f"(spread {max(vals)-min(vals):.3f}); largest single-window drop {w[1]:.3f} "
                  f"at step ~{w[0]}")
                p(f"    -> {'CLIFF (a real collapse)' if w[1] > 0.8 else 'no cliff: the ema oscillates on a plateau'}")
            else:
                p("  GROK WINDOW (after step 800): no window-over-window ema drop recorded")
    p(f"  first loss {tr[0]['loss']:.3f} -> last {tr[-1]['loss']:.3f}   "
      f"best {min(r['loss'] for r in tr):.3f}")
    return curve


def wasted_section(tr, p) -> None:
    """Steps whose sequence is over the context budget: expensive AND unlearnable.

    At the pre-W0 collate, a row with n_vis >= 4090 had its answer crushed to
    1-2 tokens and its question to 1, so a batch of them is pure noise: a
    high loss with no learnable signal, at the most expensive sequence length
    in the run.
    """
    p("\n## steps over the context budget (L > 4096: answers/questions crushed)")
    over = [r for r in tr if (r.get("L") or 0) > 4096]
    if not over:
        p("  none")
        return
    gpu = sum(r.get("step_ms", 0.0) for r in tr) / 1000.0
    t = sum(r.get("step_ms", 0.0) for r in over) / 1000.0
    p(f"  {len(over)} of {len(tr)} steps ({100*len(over)/len(tr):.1f}%)")
    p(f"  GPU time {t:,.0f} s of {gpu:,.0f} s = {100*t/max(gpu,1e-9):.1f}% of the GPU budget")
    p(f"  mean loss {st.mean(r['loss'] for r in over):.3f} vs "
      f"{st.mean(r['loss'] for r in tr if (r.get('L') or 0) <= 4096):.3f} for the rest")
    rest = [r for r in tr if (r.get("L") or 0) <= 4096]
    p(f"  mean {1000*t/len(over):,.0f} ms/step vs "
      f"{sum(r.get('step_ms', 0.0) for r in rest)/max(1, len(rest)):,.0f} ms/step")
    p("  (after the collate fix the answers survive; the rows become learnable "
      "and the cost per step falls with L^2)")


def slice_section(tr, p) -> None:
    p("\n## loss by slice (the batch loss is a token-mean over the whole batch;")
    p("  this is the batch loss on the steps where the slice holds >=50% of the ROWS --")
    p("  it is not a per-slice token-mean loss, and the two can differ)")
    if not tr or "g_mix" not in tr[0]:
        p("  NOT AVAILABLE (this log predates the g_mix field)")
        return
    p(f"  {'slice':>10} {'rows':>8} {'dom steps':>10} {'dom loss':>9} "
      f"{'first 25%':>10} {'last 25%':>9} {'trend':>8}")
    groups = sorted({g for r in tr for g in (r.get("g_mix") or {})})
    for g in groups:
        dom = [r for r in tr
               if (r.get("g_mix") or {}).get(g, 0) >= 0.5 * max(1, sum((r.get("g_mix") or {}).values()))]
        if not dom:
            continue
        q = max(1, len(dom) // 4)
        first = st.mean(r["loss"] for r in dom[:q])
        last = st.mean(r["loss"] for r in dom[-q:])
        p(f"  {g:>10} {sum((r.get('g_mix') or {}).get(g, 0) for r in tr):>8} {len(dom):>10} "
          f"{st.mean(r['loss'] for r in dom):>9.3f} {first:>10.3f} {last:>9.3f} "
          f"{last-first:>+8.3f}")
    p("  (a negative trend = the slice is learning)")


def size_section(tr, p) -> None:
    p("\n## cost by sequence length (L is the padded batch length)")
    gpu = sum(r.get("step_ms", 0.0) for r in tr) / 1000.0
    for lo, hi in ((0, 1000), (1000, 2000), (2000, 3000), (3000, 4096), (4096, 10**9)):
        w = [r for r in tr if lo < (r.get("L") or 0) <= hi]
        if not w:
            continue
        t = sum(r.get("step_ms", 0.0) for r in w) / 1000.0
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}-inf"
        p(f"  L {label:>10}  steps {len(w):>5}  loss {st.mean(r['loss'] for r in w):>6.3f}  "
          f"GPU {t:>8,.0f} s ({100*t/max(gpu,1e-9):>4.1f}% of GPU)  "
          f"mean {1000*t/len(w):>6.1f} ms/step")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--window", type=int, default=200, help="loss-curve window")
    args = ap.parse_args()

    hdr, tr, end = load_log(args.log)
    if not tr:
        print("no train records")
        return 1
    out: list[str] = []

    def p(s: str = "") -> None:
        print(s)
        out.append(s)

    p(f"# probe analysis — {args.log}")
    if hdr:
        p(f"run_id={hdr.get('run_id')}  git={str(hdr.get('git_sha'))[:10]}  "
          f"manifest_rows={hdr.get('manifest_rows')}  "
          f"batch={hdr.get('config', {}).get('batch_size')}")
    p(f"steps {tr[0]['step']}..{tr[-1]['step']}  ({len(tr)} records)")
    wall_clock_section(tr, end, p)
    loss_section(tr, args.window, p)
    slice_section(tr, p)
    size_section(tr, p)
    wasted_section(tr, p)
    if args.out:
        args.out.write_text("\n".join(out) + "\n")
        print(f"\n[analyze] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
