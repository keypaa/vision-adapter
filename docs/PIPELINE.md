# PIPELINE — reproducible rebuild manual

One honest place per stage. Each stage is a `vision_adapter` subcommand
(`python -m vision_adapter <stage>`) that shares `vision_adapter/config.py`,
`vision_adapter/manifest.py`, `vision_adapter/registry.py`, `vision_adapter/core.py`
and `vision_adapter/backends/{base,local,modal}.py`.

Historical note: the former 1800-line monolithic pipeline file has been deleted
and split into the staged modules below (see `docs/ARCHITECTURE.md` §3).

---

## 0. Prereqs

```bash
pip install torch pyarrow pillow huggingface_hub
pip install -e .            # exposes `vision-adapter` console script
# Modal backend (optional):
pip install modal && modal token new
huggingface-cli login
python -m vision_adapter --help
```

All stages log `vision_adapter/config.py:config_header` as JSONL line 0
(`run_id`, `git_sha`, `manifest_sha256`, full `TrainConfig`) and append a
`vision_adapter/registry.py:registry_entry` row to `runs.jsonl` best-effort at
`run_end` (correlated by `run_id`).

Auth chain (all stages, `backends/auth.py:15`): `--hf-token > $HF_TOKEN > $HUGGING_FACE_HUB_TOKEN > google.colab.userdata.get("HF_TOKEN")` → exported to both envs via `set_hf_token_env`. `vision_adapter/config.py:44` `get_git_sha` override: `$VISION_ADAPTER_GIT_SHA` or `$GIT_SHA` wins over `git rev-parse HEAD` (Modal has no `.git`).

**Pinned revisions** (reproducible rebuilds — commit hash not branch):

| What | Flag | Code |
|---|---|---|
| Sero + aguvis + Cauldron subset | `--upstream-pin <sha>` | `data/dataset.py:208 revision`, `data/cauldron.py:75`, `data/agentic.py:193` |
| MoonViT tower | `--revision <sha>` | `models/precompute.py:42`, `models/moonvit.py:237 hf_hub_download(..., revision=...)` |
| HF repos themselves | `MANIFEST_REPO/train_manifest.jsonl` pinned by `manifest_sha256` in config_header | `config.py:216` |

**HF repos at a glance** (code truth, not doc shorthand):

| Repo | Type | Holds |
|---|---|---|
| `keypa/MoonViT-V2-Standalone` | model | `moonvit_v2.safetensors` (`vision_tower.*`), `vision_config.json`, `moonvit.py`/`preprocess.py` code |
| `keypa/vision-adapter-manifests` | dataset | `train_manifest.jsonl` header-first + `train_manifest_val.jsonl` + `cauldron_manifest.jsonl` |
| `keypa/vision-adapter-embeddings` | dataset | `data/emb_XXXX.parquet` `103×1360=138987 rows 883.8GiB` (`stream.py:29 EMB_REPO`, `pack.py:15`) |
| `keypa/vision-adapter-grok-probe` | dataset | probe pushes `latest.safetensors+latest.opt.pt+probe_log.jsonl+probe_curves.png` each `500 steps` (`GROK_PROBE.md`) |

`[train]` optional deps `pyproject.toml:8` must include `hf_transfer` to enable `HF_HUB_ENABLE_HF_TRANSFER=1` Rust accelerator (1GiB/s agg) — without it falls back to Range (still correct, slower cold).

---

## 1. Dataset — `dataset` (30–60 min, one-off, resumable)

Builds `./data/images/{agentic,cauldron}/` + header-first
`./data/train_manifest.jsonl` (`ORDER BY image`, pinned revisions).

```bash
python -m vision_adapter dataset --out ./data --seed 0 --limit 54000
# pinned upstream revisions (reproducible rebuild):
python -m vision_adapter dataset --out ./data --upstream-pin 0xSero/glm-vision-sft-mix@<sha> --seed 0
# Modal Volume variant:
python -m vision_adapter dataset --out ./data --backend modal --seed 0
# dry-run (positional-join coverage check, no pixels):
python -m vision_adapter dataset --out ./data --dry-run
```

Verify:

```bash
head -1 ./data/train_manifest.jsonl | python -m json.tool  # {"type":"manifest_header","manifest_version":1,...}
ls ./data/images/agentic | wc -l          # ~79,659
ls ./data/images/cauldron | wc -l         # ~59k
# Modal:
modal volume ls vision-adapter-data images/agentic | head
modal volume ls vision-adapter-data metadata
```

Manifest contract: `vision_adapter/manifest.py:write_manifest_with_header`
writes line 0 header (`manifest_version`, `git_sha`, `seeds {python,numpy,torch}`,
`upstream`, `shard_set_hash`, `row_count`, `created_at`). The agentic 54k slice
uses `ORDER BY image LIMIT 54000` — without `ORDER BY`, `random.seed(0)` alone
is nondeterministic. See `docs/DATA.md`.

---

## 2. Precompute — `precompute` (10–20 min A100, ~1–2 h Colab T4, resumable)

Runs `vision_adapter/models/moonvit.py` + `vision_adapter/models/preprocess.py`
(navIT resize, BF16) and writes `embeddings/<sha1>.pt` (`[n_merged, 4096]`).

```bash
python -m vision_adapter precompute --data-dir ./data --revision <commit-sha>
# Modal A100:
python -m vision_adapter precompute --data-dir ./data --backend modal --revision <commit-sha>
# tuning:
python -m vision_adapter precompute --data-dir ./data --patch-cap 262144 --device cuda
```

Pin `keypa/MoonViT-V2-Standalone` with `--revision` (commit hash, not branch) so
a force-push cannot silently change the tower. The embedding key is
`sha1(relative_image_path)[:20].pt` relative to `images/` — identical on local
and Modal (see `vision_adapter/data/pack.py`).

Verify:

```bash
ls ./data/embeddings | wc -l                         # → 120k
modal volume ls vision-adapter-data embeddings | wc -l
python -c "import torch; print(torch.load('./data/embeddings/$(ls ./data/embeddings | head -1)', map_location='cpu').shape)"
```

---

## 3. Pack — `pack` (< 5 min, resumable)

Packs `embeddings/*.pt` into `shards/emb_XXXX.parquet`
(`SHARD_ROWS=1360`, `compression=None`, per-shard `sha256`).

```bash
python -m vision_adapter pack --data-dir ./data --shard-rows 1360
python -m vision_adapter pack --data-dir ./data --only 0:2        # range
python -m vision_adapter pack --data-dir ./data --hf-only         # HF publish
python -m vision_adapter pack --data-dir ./data --backend modal --hf-only
```

Verify:

```bash
ls ./data/shards | head
python -c "import pyarrow.parquet as pq; print(pq.read_table('./data/shards/emb_0000.parquet').num_rows)"
python -c "from vision_adapter.config import file_sha256; print(file_sha256('./data/shards/emb_0000.parquet'))"
# Modal:
modal volume ls vision-adapter-data shards | head
```

Parquet schema: `key` (`embeddings/<sha1>.pt`, byte-identical to manifest `emb`),
`n_vis` (int), `vis_bytes` (raw BF16 `tobytes()`). `file_sha256` parity is
checked post-write (Volume ↔ HF).

Shard sizes: `0.4GiB small (0-100) →8.58GiB avg →59.4GiB large (4901+), 883.8GiB total` — see `DATA.md` table. Pack Modal function (`pack.py:33`): `timeout 36000 (10h), memory 16384 MiB, volumes={"/data": vol}, secrets=[huggingface-token]` (`$HF_TOKEN`/`$HUGGING_FACE_HUB_TOKEN`), `stage_dir /var/tmp/emb_stage` (NOT `/tmp` tmpfs, ~21GiB peak). Direct FS `shutil.copyfile /data/embeddings` `50-92MB/s 42s/3.8GB` vs Volume RPC `vol.read_file_into_fileobj` `1MB/s 500s 888s tail → Worker disappeared` (`pack.py:274` with RPC fallback). Revert: `git checkout HEAD~1 -- vision_adapter/data/pack.py` restores RPC.

CLI flags (`pack.py:643`):

| Flag | Effect |
|---|---|
| `--only i[:j]` | shard range `0:2, 41:103, 101:103` (≈44s for 2 shards) |
| `--hf-only` | push to `EMB_REPO data/` only, skip `/data/shards` volume copy; `resume_action(hf_only=True)` skips if on HF alone (`pack.py:231`) |
| `--bucketed` | `6-bucket _bucket_id` sort by `n_vis` before slicing; homogeneity checked via HF `n_vis` Range not list/marker (`pack.py:418`) |
| `--shard-rows 1360` | rows per shard (changes contract) |
| `--stage-dir /var/tmp/emb_stage` | disk-backed staging |

`dataset --mix` hard-fails unless sum 100 (`dataset.py:15 _parse_mix → SystemExit(2)`): e.g. `--mix 45,45,10` OK, `50,30,10` =90 →error. `total = 0.45×total agentic slice`. `--limit` is alias of `--total` when `--mix` default; both set → `--total` wins.

---

## 4. Train — `train` / `probe` (A100-80GB or Colab T4, ~30k steps)

```bash
modal run modal_train.py::train_dryrun   # memory gate: peak < 70 GiB (legacy Volume path)
modal run modal_train.py::train
# staged CLI wrappers (config via vision_adapter/config.py:TrainConfig):
python -m vision_adapter train --data-dir ./data --config default --dryrun
python -m vision_adapter probe --data-dir ./data --max-steps 200
# HF streaming (feat/bucketed-hf-streaming, no Volume required):
HF_HUB_ENABLE_HF_TRANSFER=1 python -m vision_adapter train --data-dir ./data --max-steps 200  # Modal: whole-shard 1 GiB/s
python -m vision_adapter train --data-dir ./data --max-steps 200  # Colab T4: Range fallback (32MiB×8, prefetch)
```

Configs: `default_config()` (`bs8, log1, save200`), `probe_config()` (`bs16, log20, save500` L4), `colab_probe_config()` (`bs8, log20, save500` T4) — full 18-field table in `ARCHITECTURE.md:3a`.

**3 dry-run gates — must PASS before full train** (`modal_train.py:548-840`):

| Gate | Command | What it does | Gate file / threshold |
|---|---|---|---|
| Volume A100 | `modal run modal_train.py::train_dryrun` | 1 fwd/bwd `EmbSFT bs8` via `visual_inject`, 4 timed steps | `/data/dryrun_report.txt` `peak <70GiB` else `FAIL assert` |
| HF A100 | `modal run modal_train.py::train_hf_dryrun` | 1 step streaming via `vision_adapter/train.py:run_train(max_steps=1)` | `/tmp/hf_dryrun_report.txt` `peak <70GiB` |
| HF B300 | `modal run modal_train.py::train_hf_dryrun_b300` | all-in-VRAM `device_map {"":0}`, ckpt ON→OFF measured `step=Xs peak` | `/tmp/hf_dryrun_report.txt` `peak <250GiB` |

`train_hf` aborts `SystemExit(2)` if gate file missing or `PASS` not in it — run the dry-run first (same discipline as Volume `train_dryrun→train`).

Telemetry: `./data/logs/train_log.jsonl` (line 0 = `config_header`, last = `run_end`),
`./data/runs.jsonl`, `./data/dryrun_report.txt` — see `docs/TELEMETRY.md`. Local probe uses `probe_log.jsonl` + `probe_curves.png` + `train_curves.png` under `data_dir` (HF streaming) or `/data/logs/` (Volume).

**Probe vs Train split** (`GROK_PROBE.md` vs `modal_train.py` + `vision_adapter/train.py`):

| Aspect | Probe (Qwen3.5-2B, `grok_probe_qwen.py`/`vision_adapter/train.py:_streaming_train`) | Train (DeepSeek-V4-Flash-0731, `modal_train.py`) |
|---|---|---|
| Hidden | `2048 (2B) /2560 (4B)` | `4096` |
| Injection | `inputs_embeds`-only at `[1:1+n_vis]` (Qwen forbids ids+embeds together) | `visual_inject` hook on `embed_tokens` output (keeps `input_ids` for hash-MoE `tid2eid` routing) |
| Loss | selective `lm_head(text_hidden)` `[N,V]` only where `shift_labels != -100` (~tens tokens; full 248k vocab would be `~80GiB` at `bs16`) | full sequence via hook, same `-100` masking |
| Dtype | `bf16 Ampere+ else fp32/fp16 via scaler`, `fp32 on T4` no quant | `bf16 / FP8/FP4 MoE resident via kernels>=0.16 DeepGEMM` |
| CLI | `python -m vision_adapter probe --max-steps 200 --dtype auto/bf16/fp16/fp32` (alias `train --config colab`) | `python -m vision_adapter train --config default/probe/colab` or `modal run modal_train.py::train` |

`vision_adapter/train.py:run_train` auto-selects: has local `embeddings/*.pt` → `_local_train_with_precomputed`; else → `_streaming_train` (HF `RemoteShard` cluster sampling). `sample_size = min(len(rows), max_steps*bs*2)`, smoke `emb_0000/0001` excluded (`MAX_RG_ROWS=128`, `~9GiB single RG not streamable`).

On `feat/bucketed-hf-streaming` Modal training no longer mounts `vision-adapter-data 930 GiB` for streaming — HF `hf_transfer` whole-shard download pipelined over `33h` (`7ms/file` warm vs `4ms` Volume, `0.7%` vs `0.4%` of step). Colab Range path is bucketed by `n_vis` (`66.8%` `101-500` majority no longer pays `4900`'s cost). See `docs/DATA.md` `n_vis` distribution.

---

## Timing summary

| Stage | Wall time | Bottleneck |
|---|---|---|
| `dataset` (ETL + Cauldron + manifest) | 0.8–1.5 h + 1.5–3 h + 15–90 s | HF download |
| `precompute` (A100 / Colab T4) | 10–20 min / 1–2 h | ViT forward |
| `pack` (120k → 103 shards @1360) | < 5 min local, ~3.5 min/shard HF push | network |
| `train` (120k, bs 8, 2 epochs) | ~30k steps, ~0.1 it/s (CPU offload) | MoE backward |

Every stage is idempotent — re-running skips completed files/shards.
