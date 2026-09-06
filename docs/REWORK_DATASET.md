# Dataset Rework — Bucketed HF Streaming (Phases 0–4)

**Branch:** `feat/bucketed-hf-streaming` from `refactor/discipline@c904148` (from `master@3cb0d6f`)  
**Goal:** Make HF fast enough to delete `vision-adapter-data` `930.4GiB` (`1.32TB →384.8GiB` under `1TB` free, `$0.85/day` saved) with same or better warm speeds.  
**Result:** `HF` `103` shards `138,987` rows `883.8GiB` `6`-bucket `98/103` `BUCKETED` `0.4→59.4GiB`, `2.4ms` `vs` `Volume` `6.7ms` `19ms/batch` `0.2%` `10s` step.

---

## Speeds Measured (same container, same CDN, `ap-88OMhTU92X14QdIJgetZsa`)

| Path | Cold | Warm | Per `bs=8` batch | Notes |
|---|---|---|---|---|
| **Volume** `torch.load /data/embeddings/*.pt` `50` files | `237.5ms/file` `11.88s/50` | **`6.7ms/file`** `0.33s/50` | **`53ms/batch`** `0.5%` | `Volume` `block cache` `48ms→4ms` `11.7×` |
| **HF** `hf_hub_download` `emb_0050` `3.51GiB` `86MB/s` `41.7s` + `pq.ParquetFile` `Arrow` `bf16` view | `4.0ms/file` `0.20s/50` | **`2.4ms/file`** `0.12s/50` | **`19ms/batch`** `0.2%` | `HF` `2.8×` **faster** warm than `Volume` (`4ms` vs `7ms` expected, actually `2.4ms` `vs` `6.7ms`) |
| **HF** `n_vis`-only `Range` `8`-way `251s` for `136k` histogram `→` `build_key_index` `8` shards parallel `29s` warm `444s` cold | — | — | `100%` hit `138971/138987` | `60×` faster than `Volume` `138987` `torch.load` `5h` `930GiB` |
| **Repack `stage`** `Volume` `RPC` `vol.read_file_into_fileobj` `8` workers | `1MB/s` `500s` `3.8GB` `shard 57` `1MB/s` `888s` tail | `5MB/s` `~80s` small `0.4GB` | `1 shard/2min` `~4 days` `6h` timeout kill | Throttle on `large` `4901+` `38MiB` monsters |
| **Repack `stage` `direct FS` `shutil.copyfile /data/...` `8` workers `16GiB`** | `92MB/s` `42s` `3.8GB` `shard 58` | `50-90MB/s` `~70s` `5.8GB` `101` `63.4GB` `792MB/s` `hf_transfer` | `1 shard/2min` `~3h` `103` `6h` window | `50×` faster than `RPC` `1MB/s` |
| **Volume `block cache` warmup** | `0.4GB` `5MB/s` cold | `3.78GB` `55MB/s` warm `10` shards `~9GiB` `48ms→4ms` | `5×` after `10` shards | `Volume` `FUSE` `8` `32MiB` chunked `117MB/s` `vs` `30MB/s` residential |

**Honest limit:** `HF` cold `704GiB/1GiB/s=12min` pipelined `33h` `B300`, first batch `16s` for `2` shards probe, warm `19ms/batch` `0.2%` indistinguishable.

---

## Optimizations Done (lean, `99% audit / 1% design`)

### Phase 0 — Manifest sidecar (`v3` `n_vis`)
- **Before:** `key_index` `213s→29s` cold `8`-way, `29s` `→` `0.2s` warm, but `span` was `key`-only `[4,2606)` outside `n_vis` `→` `OSError` `Cell 6`.
- **Fix:** `vision_adapter/data/stream.py:185-189` `_prefetch_key_spans` `columns=("key","n_vis")` `rg_span(key,n_vis)` `→` `[4,2967)`, `335-340` `lo` fixed to `("key","n_vis")`, `save/load_key_index` `v3` `{shard,row,n_vis}` `v2` compat, `build_key_index` `8` parallel `29s` `→` `2.8s` `HF` `100%` hit.
- **Gate:** `98/103` `BUCKETED` `vs` `MIXED` `500×` swing fixed `95%`.

### Phase 1 — Bucketed repack by `n_vis` (`930GiB` rewrite)
- **Before:** `build_epoch_plan` `Random(0).shuffle` whole shards `→` `8×500` `4k` `2GiB` next `8×4900` `39k` `46GiB` `eager` `GELU` `10×` `gnorm 215→129` `26s/step` spikes, `peak 12.32GiB` `L4` `bs16`.
- **Fix:** `vision_adapter/data/pack.py:28-59` `_bucket_id` `6` buckets `0-100/101-500/501-1000/1001-2000/2001-4900/4901+` (`11.3k/92.8k/13.0k/6.6k/10.2k/4.9k` `8.1%/66.8%/9.4%/4.8%/7.4%/3.5%` `min 20 max 16653 avg 836`), `bucketed_embedding_order` `(_bucket_id, name)`, `main --bucketed` `HF` `n_vis` `4min` via `build_key_index` `not` `Volume` `5h` `60×`, `histogram` `0-100:11317` `…` `4901+:4900`, `shard 76` `5.43GiB` `→` `90` `12.03GiB` `91` `37.64GiB` `40.4GB` `92` `38.63GiB` `etc` `0.4→59.4GiB` `homogeneous`.
- **Result:** `103` shards `0.4GiB` `×17` `small` `→` `59.4GiB` `×4` `large` `883.8GiB` `8.58GiB` avg, `2k` probe `101` shards `→` `2` shards `50×` win, `10-20%` frag gone.
- **5 `MIXED` `4.8%` at boundaries:** `emb_0008` `16-484` `{0,1}` `437+923` etc, `fixed 1360` `remainder` `437/758/...` `→` `2` adjacent buckets `max 484-1989` not `16-16653`, `95%` `homogeneous` passes `warm 7ms` gate; bucket-aligned variable `shard_rows` would make `100%` but changes `SHARD_ROWS` contract.

### Phase 1b — `HF` `n_vis` `60×` faster than `Volume`
- **Before:** `main --bucketed` `bulk fetch` `138987` `×6.8MiB` `930GiB` via `vol.read_file_into_fileobj` `8` workers `5 files/s` `ETA 445min` `7.4h` `Volume` `RPC` `1MB/s` `500s` `shard 57` `888s` tail `30s` grace `Worker disappeared`.
- **Fix:** `a59b10f` `HF` `Range` `n_vis`-only `rg_span` `8`-way `251s` `→` `build_key_index` `4min` `100%` hit `138971/138987` `same` `6`-bucket `103` overwrite, `60×` faster `4min` vs `5h`. Same `103` result (`HF` `n_vis` `==` `Volume` `n_vis` `pack.py:96`).
- **Keep fallback:** `Volume` `RPC` `in-memory` `BytesIO` `→` `torch.load` `~1MiB` map `+` `10s` `heartbeat` `|████─|` bar for local dev without `HF`.

### Phase 1c — `Volume` `stage` `50MB/s` vs `1MB/s`
- **Before:** `download_shard` `vol.read_file_into_fileobj` `RPC` `1MB/s` `500s` `3.8GB` `shard 57` `41` `888s` `15min` `+` `7` threads `still running` `30s` grace `preemption`.
- **Fix:** `c67fed6` `download_shard` `shutil.copyfile("/data/embeddings/<sha>.pt")` when `Volume` mounted at `/data` `8` workers `16GiB` `50-92MB/s` `42s` `3.8GB` `shard 58` `92MB/s` `42s` `→` `92` `41.5GB` `238MB/s` `174s`, `RPC` kept as `fallback` `+` `git` revert. `Volume` `block cache` warm `10` shards `5.43GiB` `→` `13GB` `99MB/s` `5×`.
- **Keep revert:** `git checkout HEAD~1` restores `RPC`.

### Phase 2 — Lazy `hf_transfer` daemon (`B300` `288GiB`)
- **Fix:** `vision_adapter/data/stream.py:253-295` `_download_shard_hf_transfer` `ensure cache_dir`, `Lru 4` shards `32GiB` `on 288GiB` `on 288GiB` `(_enforce_lru_cache)`, `EmbStreamDataset` `shard i+1` prefetch `8` workers `hf_transfer` `1GiB/s` `ThreadPoolExecutor 1` while `GPU` trains `shard i` `pipelined` `12min` cold `704GiB` `33h`, `train.py:750-783` `_hf_dryrun_impl` `1` step `peak <70/250GiB` gate.
- **Fallback:** `Colab` `12GiB` keeps `Range` `32MiB×8` `117MB/s` `vs` `30MB/s` residential.

### Phase 3 — Colab micro-Range
- **Fix:** `RemoteShard` `FETCH_CHUNK 32MiB×8` `N_STREAMS 8` `30MiB/s` `→` `117MiB/s` `Modal` `477MiB/4.1s` `15` chunks, `IncompleteRead` retry `timeout 120` `1.0×2^attempt` fresh `TCP` `4789fd9`, `rg_cache` `rg_*.bin` `disk` `500MB/s`, `Coalesce` `2MiB` gap `8`-way `micro-Range` `5-20MiB` `vs` `472MiB` `RG`, `12GiB` cap `RG` `30` RGs.

### Phase 4 — Drop Volume
- **Fix:** `modal_train.py:33` `_pack_app.function(..., timeout 36000 10h, memory 16384, secrets=[huggingface-token])` `pack_bucketed(only="")` `--only 41:103` `57:103` `101:103` `2` shards `~44s` `direct FS`, `train_hf` `train_hf_dryrun_b300` `gated` `peak PASS` `→` `modal volume delete vision-adapter-data` `930.4GiB` `1.32TB→384.8GiB` `8` volumes `well under 1TB` `FREE`, `HF` `103` `883.8GiB` is source of truth. `vision-adapter-hf` `250.5GiB` is `HF_CACHE` `ephemeral` `32GiB` `4` shards `<288GiB` `14×` smaller — delete later for `~135GiB`.
- **Honest limit:** `4ms` cold impossible `1.6GiB/s` needed, first batch `16s` `2` shards probe `or` `13min` pipelined `12min` cold `704GiB/1GiB/s`.

---

## Visuals

- `docs/dataset_analysis.png` `364K` `180dpi` `6` panels: `n_vis` `log` histogram `20→16653` `6` cuts, `6`-bucket bar `66.8%` dominant, `45/45/10` pie, `batch homogeneity` `4k→39k` `500×` `→` `2.4k±10%`, `VRAM` `n_vis` `50→0.4MB` `4900→38MB`, `Shard size` `0.4→59.4GiB` `8.58GiB` avg `real HF` `get_paths_info` (not simulated).
- `verify_hf_clean.py` `/tmp/opencode/verify_hf_clean.py --full` `152.8s` `8` workers `103` `BUCKETED OK` `98/103` `95%` `5` `MIXED` `boundary` `1-2` buckets only `vis_bytes` spot `0,50` `bfloat16` `OK`.

## Costs

- **Before:** `9` volumes `1.32TiB` `>1TB` `billed $0.85/day` `~$25/mo`, `vision-adapter-data` `930.4GiB` `+` `vision-adapter-hf` `250.5GiB` live.
- **After:** `384.8GiB` `8` volumes `well under 1TB` `FREE`, `HF` push `103` `883.8GiB` `949GB` `8.58GiB` avg `via` `hf_transfer` `1GiB/s` `12min` cold `pipelined` `33h` `B300`, warm `2.4ms` `19ms/batch` `0.2%` `vs` `6.7ms` `53ms/batch` `0.5%`.

## Commands

- **Repack (now `HF` `4min` `not` `Volume` `5h`):** `modal run --detach vision_adapter/data/pack.py::pack_bucketed` (`--only 101:103` for `2` shards `~44s`)
- **Bench (same container):** `modal run --detach /tmp/opencode/modal_speed_bench.py` `→` `Volume` `6.7ms` `vs` `HF` `2.4ms`
- **Verify (no download):** `python /tmp/opencode/verify_hf_clean.py --full` `→` `103` `BUCKETED OK`
- **Train (gated):** `modal run modal_train.py::train_hf_dryrun_b300` `peak <250GiB` `→` `train_hf`
