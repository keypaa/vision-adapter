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

### RemoteShard deep dive — the numbers that matter (`stream.py:74-181, 322-351`)

* Constants: `FOOTER_BYTES=64KiB, FETCH_CHUNK=32MiB, N_STREAMS=8, MAX_RG_ROWS=128` — `emb_0000/0001` have ~9GiB single RG (`1360 rows`) → `assert biggest <=128` fails deliberately, excluded from plan.
* `rg_span(md, rgi, columns=("key","n_vis"))` uses `dictionary_page_offset` else `data_page_offset` to `+ total_compressed_size`; `_prefetch_key_spans` fetches `key+n_vis` together via `rg_span` (old bug `rg_span(key)` outside `n_vis` → `OSError Cell6`).
* `_fetch_range(url, start, end, timeout=120, retries=3)`: `IncompleteRead` retry fresh TCP `sleep 1.0×2^attempt`, other errors `0.5×2^attempt`. Disk cache `rg_{sha1(url:lo:hi)[:20]}.bin` hits skip fetch; `load_span` writes with `tmp+replace` then `_enforce_rg_cache_limit`.
* Coalesce future per-row micro-Range: `_coalesce_ranges(gap=2MiB)` merges nearby `[lo,hi]` slices (deferred until `pack.py` logs `vis_off/len`).
* `Key index v3` `{shard,row,n_vis}` per key, `v2` still loads compat (`stream.py:197-228 save/load_key_index`). Cache path `emb_cache/key_index_cache.json` or `data/cache/rg_cache`. 8-way parallel: `213s→~35s →2.8s cached`.
* LRU: Modal `_enforce_lru_cache(4 shards ≈32GiB)` on B300 288GiB ephemeral (`stream.py:279` by mtime), Colab `_enforce_rg_cache_limit(12GiB)` for `rg_*.bin` (`stream.py:322`).

### Dropping `vision-adapter-data` 930 GiB

On `feat/bucketed-hf-streaming` the training path no longer requires `modal.Volume.from_name("vision-adapter-data")` — HF streaming is the source of truth. To cut the `$0.85/day` over-`1TB` charge:

```bash
modal volume delete vision-adapter-data    # keep vision-adapter-hf (HF cache) if you want 7ms warm on next run
# or keep a minimal 64 GiB HF_CACHE volume (4 shards) for 7ms warm from job start
```

Honest limit: strict `4ms/file` cold over network is impossible (`1.6 GiB/s` needed). First batch of a fresh container pays `16s` for `2` shards (probe) or `13min` pipelined for `120k`. Keep the volume only as build-time staging for `pack.py` repack, not for training.

**Volume delete order — do not delete until Gates 1-3 + HF 103 shards verified** (`REWORK_DATASET.md:5`):

```bash
modal volume ls vision-adapter-data     # 930.4GiB — DELETE LAST, after verify_hf_clean.py 103 BUCKETED OK
modal volume ls vision-adapter-hf       # 250.5GiB HF cache 32GiB LRU ephemeral — KEEP as 64GiB warm cache or delete later (14× smaller, ~135GiB after shrink)
modal volume ls vision-graft-data       # 45.2GiB — SAFE TO DELETE (obsolete graft experiment)
# 6 other dead volumes: qwen3-cache, k3-cache, soren-*, minecraft-*, muon-output ~134.5GiB — SAFE TO DELETE
# Total: 1.32TB →384.8GiB (8 vols) well under 1TB FREE, 384→~135GiB after shrinking hf cache
```

Costs: `9 vols 1.32TiB >1TB billed $0.85/day ~$25/mo` → `384.8GiB FREE`, `HF push 103×8.58GiB 883.8GiB via hf_transfer 1GiB/s 12min cold pipelined over 33h B300, warm 2.4ms/file 19ms/batch 0.2% vs Volume 6.7ms 53ms 0.5%` (bench `modal_speed_bench.py` same container CDN `ap-88OMhTU92X14QdIJgetZsa`).

**Pack staging details** (`pack.py:33, 274-291`):

* Function: `@app.function(timeout=36000, memory=16384, secrets=[huggingface-token→HF_TOKEN])` `pack_bucketed --only 41:103 etc`.
* `stage_dir /var/tmp/emb_stage` not `/tmp` tmpfs — `~21GiB peak`, per-shard subdir `emb_XXXX.parquet.staged`.
* Download: `direct FS shutil.copyfile /data/embeddings` `50-90MB/s` (keeps RPC `vol.read_file_into_fileobj 1MB/s 500s/888s tail` as fallback, `git checkout HEAD~1` reverts). Size-checked against `vol.listdir size` map + `IOError short read` retry loop `delay 0.5×2^attempt`.
* Checkpoints surviving worker disappearance: `/data/.hf_nvis_cache` + `/data/.nvis_map_checkpoint.jsonl` + `/data/.bucketed_done` (Volume) else `/var/tmp/...`.

### 3 dry-run gates (must PASS before `train_hf`)

See `PIPELINE.md:4` table. `train_hf` checks `/tmp/hf_dryrun_report.txt` contains `PASS` else `SystemExit(2)`. Volume path still `modal_train.py:train_dryrun (EmbSFT)` with `dryrun_report.txt` on `/data`.

### Auth & env vars — the full matrix

| Var / Secret | Where | Purpose |
|---|---|---|
| `--hf-token` CLI | `backends/auth.py:get_hf_token(cli_token)` | highest priority |
| `$HF_TOKEN` | env / Modal secret `huggingface-token` / `huggingface-keypa` | `dataset/cauldron/precompute/pack/stream` pulls + pushes g, needs **write** for `--push-to-hf` (`HfApi.whoami` check) |
| `$HUGGING_FACE_HUB_TOKEN` | env fallback | same as above if `HF_TOKEN` absent |
| `google.colab.userdata.get("HF_TOKEN")` | Colab Secrets | best-effort 4th fallback |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | env | enables Rust shard download (`pyproject.toml:[train] hf_transfer`) |
| `HF_HOME=/hf`, `HF_HUB_CACHE=~/.cache/huggingface/hub` | env / `stream.py:237 _get_hf_shard_path` | HF shard lookup candidates |
| `$MODAL_TASK_ID` / `$MODAL_ENVIRONMENT` | env `stream.py:231 _in_modal()` | detects Modal vs Colab (whole-shard vs Range) |
| `$VISION_ADAPTER_GIT_SHA` / `$GIT_SHA` | env `config.py:44 get_git_sha()` | overrides `git rev-parse HEAD` when `.git` absent on Modal |

`vision_adapter/backends/modal.py:25 VOLUME_NAME="vision-adapter-data"` default; `local.py:12 root=Path`.

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
