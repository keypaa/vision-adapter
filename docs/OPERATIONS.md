# OPERATIONS — failures, retries, graceful resumption

Everything in this project is designed to be safely re-runnable. The short
version: **when in doubt, run the same command again.** This document is the
long version — the knobs, the checkpoints, the "what if it's stuck".

---

## Restart semantics per stage

| Stage | Command | Safe to Ctrl-C? | Resume behaviour |
|---|---|---|---|
| Dataset (agentic + Cauldron + manifest) | `python -m vision_adapter dataset --out ./data [--backend modal]` | yes | skips image files already in `images/agentic/`; Cauldron subsets resume at `i = 5000·k` via Volume commit every 5000 rows; manifest is deterministic given same seed + `ORDER BY image` |
| Precompute (A100) | `python -m vision_adapter precompute --data-dir ./data [--backend modal]` | yes | skips any `<hash>.pt` that already decodes to a `[n,4096]` tensor |
| Pack shards | `python -m vision_adapter pack --data-dir ./data` | yes | skips existing `emb_XXXX.parquet` shards; per-shard `sha256` verified |
| Push datasets | `python -m vision_adapter pack --data-dir ./data --hf-only` | yes | re-uploads missing or size-mismatched blobs, leaves existing ones alone |
| Dry-run gate | `modal run modal_train.py::train_dryrun` | yes | stateless: rebuilds full stack each run |
| Train | `modal run modal_train.py::train` | yes | resumes from `checkpoints/latest.pt` (see §4) |

The Colab T4 precompute variant (`vision_adapter/models/precompute.py --backend local` with a Drive-mounted `--data-dir`) has the exact same resume semantics — it hashes
`_emb_key(image_path)` to decide whether an image is already done.

## Failure modes we've seen, in order of likelihood

### 1. HuggingFace rate-limits

Symptom: `429 Client Error: Too Many Requests` during `dataset` /
`precompute` / `pack --hf-only`.

Fix: just re-run. The backend retains every completed file; nothing is wasted.
If it happens persistently, throttle the fetch loop in `precompute` by raising
`_Const.BATCH` to process fewer concurrent images, or set
`HF_HUB_ENABLE_HF_TRANSFER=0` in the image env to disable the fast path.

### 2. OOM on the A100 during `precompute`

The MoonViT is small (0.8 GB) but its activations for a 4k-patch image at
batch 16 (~64 k patches total) are sizable.  If `precompute` OOMs:

```python
# vision_adapter/models/precompute.py
class _Const:
    BATCH = 8   # was 16
```

The job is resumable so the smaller batch only costs wall time.

### 3. Memory-gate failure in `train_dryrun` (peak ≥ 70 GiB)

This means DeepSeek's activations for your sequence-length/batch don't fit on
the card next to the frozen weights. By config, this is controlled by:

| Knob | File | Default | Effect of lowering it |
|---|---|---|---|
| `batch_size` | `vision_adapter/config.py` (`TrainConfig`) | 8 | linear in activation memory |
| `max_seq_len` | `vision_adapter/config.py` (`TrainConfig`) | 4096 | linear in both activations and logits burst |
| `gpu_mem_cap_gib` | `vision_adapter/config.py` (`TrainConfig`) | 70 | the assertion threshold itself |

Change *one* at a time and rerun `train_dryrun`. The log line prints the peak
so you can see exactly how close you are.

### 4. "Training feels stuck — loss not moving for 3 000 steps"

This is expected: it's the characteristic plateau of grokking. Use
`docs/TELEMETRY.md` for the baseline; only start worrying past ~12 000 steps
at loss unchanged. The genuinely-abnormal case is the loss **climbing**,
which usually means LR too high or a corrupted embedding cache (rerun
`precompute`).

### 5. Modal session dies mid-train

`modal volume get vision-adapter-data checkpoints/` to grab `latest.pt` +
`projector_stepN.safetensors`, then the trainer reloads
`checkpoints/latest.pt` on startup (Stage 5 will pick up from there — see
`modal_train.py::train`). The full log up to the kill point is in
`logs/train_log.jsonl` for forensics (line 0 is the `config_header`).

---

## The absolute minimal watch loop

If you only remember one section, it's this. During training, open a second
shell and poll the loss every minute or two:

```bash
modal volume cat vision-adapter-data logs/train_log.jsonl | tail -5
# local:
tail -5 ./data/logs/train_log.jsonl
```

Healthy = `steps_seen` increments, `loss` hovers or drops, `peak_gib` stable.

---

## Credentials / environment

Production execution on Modal needs **two** secrets configured in the Modal
dashboard (or `modal secret create ...`):

| Secret | Used by | Purpose |
|---|---|---|
| `huggingface-keypa` (env `HF_TOKEN`) | every `@app.function` that pulls or pushes from HF | download Kimi/Google datasets + push our artefacts |
| nothing else | torch/CUDA handled by Modal | — |

Without `HF_TOKEN`, `vision_adapter/data/dataset.py` (via `agentic`/`cauldron`) will 401 on
`xlangai/aguvis-stage2` and gated fragments of `the_cauldron`.

The Colab variant needs nothing except a mounted Drive and the one-line
`from google.colab import drive` cell included in the notebook preamble.
