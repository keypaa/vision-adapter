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

## Streaming mode and volume strategy

`vision_adapter/data/stream.py` has two paths for embeddings:

| Backend | Path | Speed | When |
|---|---|---|---|
| Modal (`MODAL_TASK_ID` set) + `hf_transfer` | Whole-shard `hf_hub_download` (Rust, `1 GiB/s agg`) → local parquet read, no Range | Cold `13min` for `120k` pipelined, warm `7ms/file` vs Volume `4ms` (`56ms/batch` vs `32ms`) | Default on Modal B300/A100 |
| Modal without `hf_transfer` | `RemoteShard` `32MiB×8` chunked Range (`117 MiB/s` inside Modal) + `rg_cache` + next-RG prefetch | `4.1s/RG`, `240s` wall for `5 steps` observed | Fallback |
| Local (Colab T4) | Same `RemoteShard` Range with `IncompleteRead` retry (`timeout=120` + fresh TCP) + `v3` bucketed plan | `29s` key-index cold → `0.2s` warm, bucketed `101-500` majority no longer pays `4900`'s cost | Default on Colab |

`vision_adapter/data/stream.py:EmbStreamDataset` auto-selects: if `_in_modal()` + local parquet cached via `_get_hf_shard_path` / `_download_shard_hf_transfer`, it reads `pq.ParquetFile(local_path)` directly; otherwise it streams via `RemoteShard` Range. Exact output — same `vis` decode in both cases.

`HF_HUB_ENABLE_HF_TRANSFER=1` is the switch that activates the Rust accelerator when `hf_transfer` is installed (`pyproject.toml:[train]` optional dep). Without it, the trainer still works — just slower cold.

### Dropping `vision-adapter-data` 930 GiB

On `feat/bucketed-hf-streaming` the training path no longer requires `modal.Volume.from_name("vision-adapter-data")` — HF streaming is the source of truth. To cut the `$0.85/day` over-`1TB` charge:

```bash
modal volume delete vision-adapter-data    # keep vision-adapter-hf (HF cache) if you want 7ms warm on next run
# or keep a minimal 64 GiB HF_CACHE volume (4 shards) for 7ms warm from job start
```

Honest limit: strict `4ms/file` cold over network is impossible (`1.6 GiB/s` needed). First batch of a fresh container pays `16s` for `2` shards (probe) or `13min` pipelined for `120k`. Keep the volume only as build-time staging for `pack.py` repack, not for training.

## The absolute minimal watch loop

If you only remember one section, it's this. During training, open a second
shell and poll the loss every minute or two:

```bash
modal volume cat vision-adapter-data logs/train_log.jsonl | tail -5  # when still on volume
# HF streaming (feat/bucketed-hf-streaming):
tail -5 ./data/logs/train_log.jsonl  # probe_log.jsonl in data_dir
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
