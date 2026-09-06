# QUICKSTART — from zero to a trained projector

> Clone: `git clone https://github.com/keypaa/vision-adapter.git && cd vision-adapter` — note: folder is **lowercase** `vision-adapter`, not `Vision-Adapter`.

This is the end-to-end, copy-pasteable path. Each step lists what it does, how
long it takes, and how to verify it succeeded. The default path is **local
only** (no HF push). Add `--push-to-hf --hf-repo ...` when you explicitly want
to publish. Any GPU works (T4, L4, A100, 4090) — `dataset/pack` run on CPU,
`precompute/train` require GPU.

---

## Prereqs (one-time)

```bash
git clone https://github.com/keypaa/vision-adapter.git
cd vision-adapter
pip install -e .            # base: torch, pyarrow, pillow
pip install -e ".[train]"   # training: transformers, safetensors, accelerate, sentencepiece
# optional for Modal backend:
pip install modal
modal token new            # opens browser; links your Modal account
# HF CLI already logged in? If not:
#   huggingface-cli login
```

Check your HF identity (needed to push to HF or to get higher rate limits). For
read-only pulls you can stay anonymous; for pushes you need a **write** token:

```bash
python3 -c "from huggingface_hub import whoami; print(whoami())"
# or export HF_TOKEN (or pass --hf-token) — see note below
```

> **HF auth (4-level chain, `backends/auth.py:15`):** `--hf-token HF_xxx` > `$HF_TOKEN` > `$HUGGING_FACE_HUB_TOKEN` > `google.colab.userdata.get("HF_TOKEN")` → exported to both envs via `set_hf_token_env` so `huggingface_hub/datasets` pick it up. Anonymous works for pulls but is slower (lower rate limits, `Warning: unauthenticated`). For `—push-to-hf` you need a **write** token — the CLI checks `HfApi.whoami()` and refuses a read-only token with `read-only — create a write token`. On Modal the secret is `huggingface-token` → `$HF_TOKEN`. `$VISION_ADAPTER_GIT_SHA`/`$GIT_SHA` overrides `git SHA` when `.git` absent. `HF_HUB_ENABLE_HF_TRANSFER=1` needs `pip install -e .[train]` (`hf_transfer` dep) for Rust 1GiB/s else Range fallback.

---

## Step 0 — Projector source: agentic images + MoonViT weights

These are already done and reused; you only re-run if you change something.

* `vision_adapter/data/extract_moonvit_v2.py` — pulled the 401M MoonViT-V2 weights + config out of
  `moonshotai/Kimi-K3` and pushed them to `keypa/MoonViT-V2-Standalone`.
  *Re-run only if you delete the HF repo.*
* `python -m vision_adapter dataset --out ./data --total 20000 --dry-run` — verifies the six positional joins are
  still intact; safe, touches no pixels.
  *Re-run before any `dataset` if the upstream datasets update.*

---

## Step 1 — Dataset: build images + header-first manifest  (≈ 30–60 min, one-off)

Downloads the sources (≈ 23 GB), runs the resize/pad pipeline, writes
`./data/images/{agentic,cauldron}/` + `./data/train_manifest.jsonl` (header-first,
`ORDER BY image`, pinned revisions via `vision_adapter/data/dataset.py`).

**Why 54k?** `120k total × 45% agentic ≈ 54k` — agentic is 45% of the
`120k = 45% agentic / 45% doc / 10% conv` mix (`45+45+10=100`). The pool is
`0xSero/... 82,829` rows; `LIMIT 54000` is the balanced slice. Use `--total`
to get a smaller dataset that **keeps the ratio** (e.g. `--total 20000 →
9k/9k/2k`).

```bash
# 120k production (54k / 54k / 12k) — default 45,45,10 (local only, no push):
python -m vision_adapter dataset --out ./data --seed 0
# 20k probe that keeps the mix (9k agentic / 9k doc / 2k conv):
python -m vision_adapter dataset --out ./data --seed 0 --total 20000 --mix 45,45,10
# hardblock: must sum to 100 — this exits 2:
python -m vision_adapter dataset --out ./data --mix 50,30,10  # 90 != 100 → error
# custom mix that still sums to 100:
python -m vision_adapter dataset --out ./data --total 20000 --mix 50,30,20
# Modal Volume variant:
python -m vision_adapter dataset --out ./data --backend modal --seed 0
# dry-run (positional-join coverage check, no pixels):
python -m vision_adapter dataset --out ./data --total 20000 --dry-run
# legacy single-group compat (--limit is alias of --total when --mix is default):
python -m vision_adapter dataset --out ./data --limit 100 --dry-run
# publish to a HF dataset repo (opt-in — local by default; write token required):
python -m vision_adapter dataset --out ./data --total 120000 --mix 45,45,10 --push-to-hf --hf-repo keypa/vision-adapter-manifests --hf-token $HF_TOKEN
# or via env: HF_TOKEN=$HF_TOKEN python -m vision_adapter dataset --out ./data --push-to-hf --hf-repo ...
```

Verify (local):

```bash
ls ./data/images/agentic | head
head -1 ./data/train_manifest.jsonl | python -m json.tool   # {"type":"manifest_header",...}
# mix provenance is in the header tags:
python -c "import json; h=json.loads(open('./data/train_manifest.jsonl').readline()); print(h['tags'])"
# {"limit":20000,"total":20000,"mix":"45,45,10","agentic_slice":9000,"provenance_note":"ORDER BY image"}
```

Or on Modal:

```bash
modal volume ls vision-adapter-data images/agentic | head
modal volume ls vision-adapter-data metadata
```

Expect `images/agentic` to mirror the chosen mix (e.g. `9k` PNGs for `--total 20000`).

**Push vs local:** `dataset` stays **local by default** — no HF write unless you add
`--push-to-hf --hf-repo <ns/repo> --hf-token <write-token>`. The CLI checks that
the token is a **write** token via `HfApi.whoami()` before uploading
`train_manifest.jsonl` (manifest repo) + `dataset_info.json` + provider card.
The embedding shards (`shards/emb_*.parquet`) are pushed separately via Step 3
(`pack --hf-only`) to the **embeddings** repo — different repo, different card.
See `docs/PIPELINE.md` and `docs/HF_PUBLISH.md`.

---

## Step 2 — Precompute MoonViT embeddings (choose ONE)

Embeddings are per-image `[n_merged, 4096]` BF16 tensors, hashed
`sha1(relative_image_path)[:20].pt`. Same hash convention on both backends, so
results are interchangeable.

### 2a · Modal A100 (fast, ~10–20 min)

```bash
python -m vision_adapter precompute --data-dir ./data --backend modal
```

### 2b · Free Colab T4 (≈ 1–2 h, resumable across sessions)

1. Upload `images/` to Drive: `MyDrive/vision_adapter/images/`
2. Open `vision_adapter/models/precompute.py` (Colab path uses `--backend local` with a Drive-mounted `--data-dir`; see `vision_adapter/models/precompute_colab.py` — deprecated shim that re-exports the shared precompute), run cells 1 → 3.
3. When done, sync the embedding folder back to Modal. The safest way is to `zip`
   it first (79 659 files, ~6 GB) and unzip inside the Modal Volume:

```bash
cd /content/drive/MyDrive/vision_adapter/embeddings
zip -r /content/drive/MyDrive/vision_adapter/embeddings.zip .
```

Then locally:

```bash
# download embeddings.zip from Drive to ~/Downloads/
mkdir -p /tmp/opencode/embeddings_push
cd /tmp/opencode/embeddings_push && unzip ~/Downloads/embeddings.zip
modal volume put vision-adapter-data ./. /embeddings/
```

4. On Modal, one-line sanity check that both precompute backends produced the
   same filename scheme (each embedding is a valid .pt of shape [n,4096]):

```bash
modal volume ls vision-adapter-data embeddings | head
```

Verify either way (local or Modal):

```bash
python -m vision_adapter precompute --data-dir ./data --help   # shows --revision pin, --patch-cap
ls ./data/embeddings | wc -l            # local: should approach 120k (or 20k for probe)
modal volume ls vision-adapter-data embeddings | wc -l   # Modal: should approach 120k
```

> **Colab hint:** `dataset`/`pack` run on CPU, but `precompute`/`train` require
> **any** CUDA GPU (T4, L4, A100, 4090…). On CPU you get:
> `[vision-adapter] 'precompute' requires a CUDA GPU — nvidia-smi not found`.

---

## Step 3 — Pack embeddings into shards  (< 5 min local, ~3h for 103 via HF, resumable)

Packs the `.pt` embeddings into `103 shards SHARD_ROWS=1360, compression=None, 883.8GiB total 8.58GiB avg 0.4→59.4GiB` (`vision_adapter/data/pack.py` schema `key/n_vis/vis_bytes` bf16→tobytes, per-shard `sha256`, `shard emb_XXXX.parquet`).

```bash
python -m vision_adapter pack --data-dir ./data
# HF-only push (no local volume copy):
python -m vision_adapter pack --data-dir ./data --hf-only --hf-token $HF_TOKEN
# single-shard range / resume:
python -m vision_adapter pack --data-dir ./data --only 0:2
python -m vision_adapter pack --data-dir ./data --only 41:103  # resume after kill at 41 (888s stall)
# bucketed repack (fixes 500× swing, 6-bucket 0-100/…/4901+):
python -m vision_adapter pack --data-dir ./data --bucketed --hf-only --hf-token $HF_TOKEN  # HF n_vis 4min vs Volume 5h 60×
# Modal bucketed (production, direct FS 50-92MB/s 42s/3.8GB vs RPC 1MB/s 500s):
modal run vision_adapter/data/pack.py::pack_bucketed --detach          # all 103
modal run vision_adapter/data/pack.py::pack_bucketed -- --only 101:103  # 2 shards ~44s
```

Flags: `--only i[:j]` shard range, `--hf-only` skip `/data/shards` volume copy (`resume_action hf_only`), `--bucketed` sort `(_bucket_id(n_vis), name)` then homogeneity check via HF `n_vis` Range (`98/103 BUCKETED`), `--stage-dir /var/tmp/emb_stage` not `/tmp` tmpfs (~21GiB peak). Modal function `pack_bucketed` `timeout 36000 (10h) memory 16384` direct FS fallback `git checkout HEAD~1` for RPC revert. `dataset --mix 45,45,10` hard-fails `SystemExit(2)` if sum≠100.

Verify:

```bash
ls ./data/shards | head
python -c "import pyarrow.parquet as pq; print(pq.read_table('./data/shards/emb_0000.parquet').num_rows)"
python -c "from vision_adapter.config import file_sha256; print(file_sha256('./data/shards/emb_0000.parquet'))"
# bucketed verify (no download, 152s 8 workers):
python /tmp/opencode/verify_hf_clean.py --full  # 103 BUCKETED OK 98/103 95%
```

---

## Step 4 — Training (A100-80GB, or Colab T4 probe)

**Dry-run gates first — must PASS before full train** (3 gates, see `PIPELINE.md:4`):

```bash
modal run modal_train.py::train_dryrun            # Volume A100: peak <70GiB → PASS → /data/dryrun_report.txt
modal run modal_train.py::train_hf_dryrun          # HF A100: peak <70GiB → /tmp/hf_dryrun_report.txt
modal run modal_train.py::train_hf_dryrun_b300     # HF B300: peak <250GiB all-in-VRAM → PASS then train_hf
# local staged validation (CPU-safe, no GPU):
python -m vision_adapter train --data-dir ./data --dryrun            # validates manifest+config+backend no kernels
python -m vision_adapter train --data-dir ./data --dryrun --config probe  # bs16
```

Expected tail:

```
[dryrun] loss=… n_trainable=67.1M | mem_alloc=…GiB peak=…GiB budget=70GiB -> PASS
[hf-dryrun] rc=0 mem_alloc=… peak=… budget=70GiB -> PASS
```

(`train_hf` aborts `SystemExit(2)` if `/tmp/hf_dryrun_report.txt` missing or not `PASS` — same discipline as Volume.)

**Probe vs Train** (`GROK_PROBE.md` vs `modal_train.py`): Probe Qwen3.5-2B `H2048, inputs_embeds-only [1:1+n_vis], selective lm_head loss tens tokens (248k vocab ~80GiB avoided)` vs Train DeepSeek-V4 `H4096, visual_inject hook keeping input_ids for MoE hash gate, FP8/FP4 via kernels>=0.16`. `vision_adapter/train.py:run_train` auto-selects local `embeddings/*.pt` → `_local_train_with_precomputed` else HF streaming cluster-sampled `EmbStreamDataset` (`MAX_RG_ROWS=128`, smoke `emb_0000/0001 ~9GiB excluded`). `python -m vision_adapter probe` is alias for `train --config colab`.

**Then start training.** 2 epochs × (120 000 / 8) ≈ **30 000 steps** (for a
`20k` probe it's ~`5k` steps). Grokking is sample-bound, not step-bound: watch
`samples_seen` — the Baseten cliff (≈ step 900 at bs 64) maps to
**≈ step 7 200–11 000 at our bs 8**. Loss curve is printed every 20 steps and
the full JSON telemetry stream lands in `/data/logs/train_log.jsonl`
(see `docs/TELEMETRY.md`):

```bash
modal run modal_train.py::train                 # Volume A100 fallback
modal run modal_train.py::train_hf              # HF streaming B300 (gated on train_hf_dryrun_b300)
modal run modal_train.py::train_hf_a100         # HF streaming A100 (gated on train_hf_dryrun)
# staged local (any GPU):
python -m vision_adapter train --data-dir ./data --config probe --max-steps 200
python -m vision_adapter train --data-dir ./data --max-steps 200 --dtype auto/bf16/fp16/fp32  # T4 fp32 else bf16
python -m vision_adapter probe --data-dir ./data --max-steps 200 --hf-token $HF_TOKEN
```

Checkpoints land in `/data/checkpoints/projector_step*.safetensors` and
`latest.pt` every 20 steps (loss-tracked). For probe runs: `probe_log.jsonl` (HF streaming) or `train_log.jsonl` (Volume) + `probe_curves.png`/`train_curves.png` + `runs.jsonl` (see `TELEMETRY.md` file locations).

To fetch the trained adapter locally after the run:

```bash
modal volume get vision-adapter-data checkpoints/ ./checkpoints/
```

---

## Step 5 — Evaluate

Quick smoke CPU test — run a single image through the frozen MoonViT and the
trained projector, eyeball the shapes:

```python
import torch, json
from PIL import Image
from safetensors.torch import load_file
from vision_adapter.models.moonvit import load_moonvit_from_safetensors
from vision_adapter.models.preprocess import process_image

cfg = json.load(open('.cache/vision_config.json'))
vit = load_moonvit_from_safetensors('.cache/moonvit_v2.safetensors', cfg,
                                    device='cpu', dtype=torch.float32)
emb = vit(**process_image(Image.new('RGB', (280, 280))))[0]
proj = load_file('checkpoints/projector_final.safetensors')
print('vit merged tokens:', emb.shape)   # expect [n, 4, 1024]
# then run projector + frozen DeepSeek from modal_train.
```

Full MMMU/UI-grounding eval lives outside this repo — this project only trains
the alignment layer.

---

## Troubleshooting — expanded (`OPERATIONS.md:22`)

| Symptom | Fix |
|---|---|
| `MEMORY GATE FAIL` on dryrun | batch_size 8 too big for your Modal A100 (70GiB) or B300 (250GiB) SKU → lower `batch_size` in `vision_adapter/config.py` (`TrainConfig`) or `max_seq_len 4096` |
| `hf-dryrun` abort `SystemExit(2) PASS not found` | you skipped `train_hf_dryrun` — run `modal run modal_train.py::train_hf_dryrun_b300` first then `train_hf` checks `/tmp/hf_dryrun_report.txt` |
| loss stays ~7 after step 12 000 | grokking may not happen this run; try `LR = 7e-4` and rerun; see `docs/TELEMETRY.md` |
| Colab precompute synced but trainer sees no images | `embeddings/` must be at Volume root: `modal volume put vision-adapter-data ./embeddings/. /embeddings/` |
| Out of RAM in ETL | Cauldron pulls whole configs; raise `memory=` on the `dataset` stage (Modal) |
| Colab session died | just re-run the precompute cell; `_already_done()` skips everything already cached (sha1[:20] key) |
| `KeyError` in aguvis join | upstream re-indexed; re-run `python -m vision_adapter dataset --dry-run` (fail-closed `IndexError` verifies 6 prefixes) |
| `IncompleteRead` mid-Range | HF CDN truncated mid-chunk — `stream.py:50 _fetch_range timeout120 fresh TCP 1.0×2^attempt` auto-retries 3 times |
| `Worker disappeared` after shard 57 888s tail | Volume RPC throttle `1MB/s` (vs direct FS `50-90MB/s`) — `pack.py:274` already uses `shutil.copyfile /data/...` with RPC fallback; resume `pack_bucketed only=57:103` and checkpoint survives at `/data/.bucketed_done` |
| `short read expected != bytes` | stream closed clean but short — `pack.py:283,296` size check vs `vol.listdir size` map + retry loop `0.5×2^attempt` |
| `read at X outside span [lo,hi)` | old bug `rg_span(key)` outside `n_vis span` — fixed `stream.py:335` to `columns=("key","n_vis")` |
| `Volume 1MB/s` still after pack fix | check container started before fix — `git checkout HEAD~1` restores RPC; new image uses direct FS `92MB/s 42s/3.8GB` |

If in doubt, re-run the failing step — every stage is idempotent. Pack `bucketed` keeps `/data/.hf_nvis_cache` + `/data/.nvis_map_checkpoint.jsonl` so rebuild after kill resumes not restarts.

## Colab local GPU notes

`dataset` and `pack` run on CPU. `precompute` and `train` require **any**
CUDA GPU (`has_gpu()` checks `torch.cuda.is_available() || nvidia-smi` —
not just T4). On CPU you get:

```
[vision-adapter] 'precompute' requires a CUDA GPU, but none was detected
Any CUDA-capable GPU works (T4, L4, A100, 4090…) — no SKU requirement.
```

For `train` use `--dryrun` for a CPU probe (manifest + config + backend
listed, no kernels, `dryrun ok`).
