#!/usr/bin/env python3
"""Beautiful probe graphs with plotly, linear scales, no matplotlib."""
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# find latest probe log
candidates = [
    Path("logs/probe_l4_200_probe_log.jsonl"),
    Path("data/probe_log.jsonl"),
    Path("logs/heldout60_evolution.json"),
]
probe_path = next((p for p in candidates if p.exists()), None)
if probe_path is None:
    # try molab 1000
    probe_path = Path("checkpoints/molab_1000/probe_log.jsonl") if Path("checkpoints/molab_1000/probe_log.jsonl").exists() else None

if probe_path is None:
    import glob
    logs = sorted(Path("logs").glob("probe*.jsonl"))
    probe_path = logs[-1] if logs else None

if probe_path is None or not probe_path.exists():
    print(f"no probe log found, tried {candidates}")
    raise SystemExit(1)

print(f"using {probe_path} {probe_path.stat().st_size/1024:.1f}KB")
recs = []
with open(probe_path) as f:
    for line in f:
        try:
            r=json.loads(line)
        except: continue
        if r.get("type")=="train" and "step" in r:
            recs.append(r)
        elif "loss" in r and "step" in r:
            recs.append(r)

# also try heldout 1000 molab if exists
molab_log = Path("data/probe_log.jsonl")
if molab_log.exists() and molab_log != probe_path:
    print(f"also found molab {molab_log}")

steps = [r["step"] for r in recs]
loss = [r["loss"] for r in recs]
ema = [r.get("ema_loss", r.get("loss_ema", r["loss"])) for r in recs]
gnorm = [r.get("gnorm", r.get("grad_norm", 0)) for r in recs]
lr = [r.get("lr", 0) for r in recs]
samples = [r.get("samples_seen", r["step"]*16) for r in recs]

print(f"steps {len(steps)} {steps[0]}->{steps[-1]} loss {loss[0]:.2f}->{loss[-1]:.2f} ema {ema[0]:.2f}->{ema[-1]:.2f}")

# Beautiful layout: 3 rows, linear only
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("Loss — raw vs EMA (linear)", "Gradient Norm (linear)", "Learning Rate"))

# Row 1: loss
fig.add_trace(go.Scatter(x=steps, y=loss, mode="lines", name="raw", line=dict(color="#8ab4f8", width=1), opacity=0.6), row=1, col=1)
fig.add_trace(go.Scatter(x=steps, y=ema, mode="lines", name="EMA .98", line=dict(color="#1a73e8", width=2.5)), row=1, col=1)
# plateau shading 350-570 for 200, or 350-1000 for 1000
if max(steps) >= 570:
    fig.add_vrect(x0=350, x1=max(steps), fillcolor="orange", opacity=0.08, line_width=0, row=1, col=1)
    fig.add_annotation(x=450, y=max(ema)*0.9, text="Plateau / Grok<br>(Expected flat)", showarrow=False, bgcolor="#fff3cd", bordercolor="#ff9800", row=1, col=1)

# Row 2: gnorm
fig.add_trace(go.Scatter(x=steps, y=gnorm, mode="lines", name="grad norm", line=dict(color="#a78bfa", width=1.2)), row=2, col=1)

# Row 3: lr
fig.add_trace(go.Scatter(x=steps, y=lr, mode="lines", name="lr", line=dict(color="#fbbf24", width=1.5)), row=3, col=1)

fig.update_xaxes(title_text="steps", row=3, col=1)
fig.update_yaxes(title_text="loss", row=1, col=1)
fig.update_yaxes(title_text="gNorm", row=2, col=1)
fig.update_yaxes(title_text="lr", row=3, col=1)

fig.update_layout(height=900, width=1100, title_text=f"Vision Adapter Probe — {probe_path.name} — Steps {steps[0]}-{steps[-1]} — linear scales", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
# secondary x for samples_seen on top
# Add annotation for samples
fig.add_annotation(text=f"samples_seen {samples[0]} → {samples[-1]} ({samples[-1]/1000:.1f}k)", xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False, font=dict(size=12, color="#5f6368"))

out_html = Path("logs/probe_beautiful.html")
fig.write_html(str(out_html))
print(f"wrote {out_html} {out_html.stat().st_size/1024:.1f}KB")

# also try to write png if kaleido available
try:
    out_png = Path("logs/probe_beautiful.png")
    fig.write_image(str(out_png), width=1100, height=900, scale=2)
    print(f"wrote {out_png} {out_png.stat().st_size/1024:.1f}KB")
except Exception as e:
    print(f"png export skipped (need kaleido): {e}")
    # fallback: try to save as static via plotly
    pass

# Also create a simple standalone html with no log scale
print("done — open logs/probe_beautiful.html in Marimo or browser")
