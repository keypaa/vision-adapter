# TELEMETRY — what to watch when you can't watch it live

The trainer writes a single JSONL stream to `./data/logs/train_log.jsonl`
(and `/data/logs/train_log.jsonl` on Modal — one object per step). After the
run — or mid-run from another shell — pull it:

```bash
modal volume get vision-adapter-data logs/ ./logs/
# local:
cat ./data/logs/train_log.jsonl | tail -5
```

Line 0 is always the `config_header` (`vision_adapter/config.py:config_header`):
`{"type":"config_header","run_id":...,"git_sha":...,"manifest_sha256":...,"manifest_rows":...,"config":{...}}`.
The run's closing `run_end` and the `runs.jsonl` registry entry
(`vision_adapter/registry.py:registry_entry`) are correlated by `run_id`.

## Keys you'll see

### config header (`"type": "config_header"` — line 0)

| Field | Meaning |
|---|---|
| `run_id` | short run identifier (`YYYYMMDDTHHMMSSZ-xxxxxx`) |
| `git_sha` | `git rev-parse HEAD` at run start |
| `manifest_sha256` / `manifest_rows` | hash + row count of the manifest used |
| `config` | verbatim `TrainConfig` dict (every field) |
| `timestamp` / `python` / `platform` | provenance |

### train step (`"type": "train"`)

| Field | Meaning |
|---|---|
| `step` | optimizer step counter (1-based across both epochs) |
| `epoch` | 0 or 1 |
| `loss` | cross-entropy on answer tokens only (prompt tokens masked) |
| `samples_seen` | cumulative samples = step × batch_size(8) |
| `lr` | current LR (constant 5e-4 unless you add a scheduler) |
| `peak_gib` | `torch.cuda.max_memory_allocated()` at this point — leak detector |
| `it_s` | rolling steps/sec since run start |
| `ts` | unix timestamp |

### val probe (`"type": "val"`)

Every 250th step. Held-out `train_manifest_val.jsonl` — never optimised.

| Field | Meaning |
|---|---|
| `val_loss` | mean CE on the data never seen by the optimiser |
| `n_rows`   | how many held-out examples were averaged |

### run end (`"type": "run_end"` — last line)

Written best-effort at trainer exit with `run_id`, `wall_min`, `final_loss`,
`peak_gib`, and `step_ms` breakdown — mirrors the `runs.jsonl` registry row.

## The one curve that matters in this project

```
train loss
10 |╮
   | ╰╮                                       ← plateau (projector not yet aligned)
 5 |   ╰───────────────╮
   |                   ╰──╮                   ← grokking cliff (~step 7-11k)
 2 |                      ╰───────────────    ← post-grok convergence
   └────────────────────────────► step
   0   2k   4k   6k   8k  10k  12k  14k
```

Baseten's GLM recipe collapsed at ~step 900 (batch 64, 66k img = ~58k samples).
We have the same objective at batch 8 — watch `samples_seen` for the equivalent
~58k samples (≈ step 7 200 at batch 8), and expect the collapse anywhere in the
7 k–11 k range. **Do not kill the run** at step 900 expecting the same cliff.

## What "good" looks like vs "something is wrong"

| Observation | Verdict |
|---|---|
| `peak_gib` flat across epochs | healthy |
| `peak_gib` ratchets up slowly | memory leak → kill, file bug, restart from `latest.pt` |
| `val_loss` ≈ `loss` before grok | normal, projector output is noise for both |
| `val_loss` diverges from `loss` *after* grok | overfit on the mix; bump up data diversity |
| `loss` never collapses by ~epoch 1.5 | either: (a) embedding cache corrupted (rerun `precompute`), or (b) LR too low → try 7e-4 |
| `it_s` ~ 0.1 | expected. CPU offloading of a 155 GiB MoE is PCIe-bound. Baseten used 8 GPUs; we don't. |

## On-disk artefacts you can diff between runs

```
./data/logs/train_log.jsonl             # the stream (line 0 = config_header)
./data/runs.jsonl                       # registry: one JSON row per run (vision_adapter/registry.py)
./data/dryrun_report.txt                # one-line peak-memory verdict
./data/checkpoints/projector_step*.safetensors
./data/checkpoints/projector_final.safetensors
```

Load a projector locally (no GPU needed):

```python
from safetensors.torch import load_file
proj_sd = load_file("checkpoints/projector_final.safetensors")
print({k: tuple(v.shape) for k, v in proj_sd.items()})
```
