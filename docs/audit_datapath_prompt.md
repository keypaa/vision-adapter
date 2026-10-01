# Data-path audit prompt — Vision-Adapter, fresh session

Copy everything below into a new session with no prior context.

---

You are auditing the **data loading path** of a vision→LLM adapter trainer at
`/home/keypaa/Projects/DSV4-0731/Vision-Adapter`. Nothing else. The training
maths, the projector, the geometry and the loss were audited separately on
2026-09-30 and are out of scope here.

## The symptom to explain

A 2-step smoke test (`--max-steps 2 --batch-size 2`) on a Colab T4 ran for
**~30 minutes and was then killed by the OOM killer**, without reaching the
training loop. Its `sample_size` was 8 rows. The kill happened while
streaming embedding shards, not while touching the GPU.

Representative tail of that log:

```
[stream] load_span emb_0094.parquet 1842MiB @1886776429
[stream] load_span emb_0094.parquet 1818MiB @13692253849
[stream] prefetched emb_0094.parquet rg7 (1818MiB in 1172s)
[stream] RG cache evicted rg_29d2be119d110d6ef055.bin (21MiB, cap 12GiB)
... 13 more evictions ...
[stream] streamed emb_0094.parquet rg1 (1842MiB in 1178s)
Killed
```

Other shards in the same run show ~170 MiB row groups; `emb_0094` shows
**1.8 GiB** ones. One prefetch took 1172 s.

## Shard sizes are not uniform — they span 146x

From the repo listing (`keypa/vision-adapter-embeddings`, 103 shards, 949 GB
on disk / ~832 GiB of parquet). Every shard holds 1360 rows; the row *byte*
size varies because `n_vis` varies per image:

| shards | parquet size | implied mean `n_vis` | 128-row group |
|---|---|---|---|
| `emb_0000`-`0007` (8) | ~435 MB | ~35 | 0.03 GiB |
| `emb_0008`-`0075` (68) | ~3.8 GB | ~350 | 0.34 GiB |
| `emb_0076`-`0085` (10) | 5.8-8.4 GB | ~700-900 | 0.7-0.9 GiB |
| `emb_0086`-`0090` (5) | 12-13 GB | ~1,240 | 1.2 GiB |
| `emb_0091`-`0097` (7) | 40-42 GB | ~4,000 | 3.9 GiB |
| `emb_0098`-`0101` (4) | 50-64 GB | ~5,800 | 5.7-6.0 GiB |
| `emb_0102` (1) | 12.8 GB | — | — |

Three things follow, and they change what is worth investigating:

1. **The 27 largest shards hold ~83% of the corpus.** `emb_0099` is 146x
   `emb_0000`.
2. **They look sorted ascending.** `0000`-`0007` at 435 MB, then a jump to
   3.8 GB, then 8, then 40, then 63 — consistent with a packer that wrote
   row groups smallest-first. If that is true, shard index is a proxy for
   size, which makes "which shards are in this plan" a cheap thing to reason
   about. Verify it rather than assuming.
3. **`MAX_RG_ROWS = 128` cannot protect the large shards.** The guard is on
   rows; the cost is `128 x n_vis x 4096 x 2` bytes. A 6 GiB group passes a
   128-row assert unchanged, and with the prefetch thread plus the main
   thread both holding a span, plus the fp32 collate copy, the theoretical
   peak on the largest shard is well past what a 12 GB Colab has.

**An inconsistency worth resolving, not assuming.** The table above predicts
128-row groups for `emb_0094` (40-42 GB shards) of ~3.9 GiB, but the log
reports **1.8 GiB** for that shard. Either `emb_0094` is packed with smaller
row groups than the others, or its mean `n_vis` is lower than shard size
alone implies (shard size depends on compression, not only on raw
`n_vis x 4096 x 2`). Get the real row-group layout from the parquet footer
rather than deriving it — `RemoteShard` already fetches that footer, so a
loop over the 103 shards printing `num_rows`, `total_byte_size` and
`total_compressed_size` per group would settle it in one pass.

`build_epoch_plan` shuffles shards before selecting, so whether a run meets a
0.03 GiB shard or a 6 GiB one is currently luck. Nothing in the code warns.

## What the path is supposed to do

`vision_adapter/data/stream.py`:

- `list_shards()` — enumerate the 103 parquet shards on `keypa/vision-adapter-embeddings`.
- `build_key_index()` — build `emb key -> (shard, row_idx, n_vis)` by Range-fetching
  only the `key` and `n_vis` columns. Cached to `key_index_cache.<hash>.json`
  (~13 MB, ~2-5 min cold, 0.2 s warm).
- `build_epoch_plan()` — pick rows for a run, bucketed by `n_vis`, whole shards.
- `EmbStreamDataset.__iter__` — walk the plan, fetch the containing row-group,
  yield collate-ready rows.
- `RemoteShard` — an `io.RawIOBase` serving only a 64 KiB footer and the
  currently-prefetched span over HTTPS Range; `load_span(lo, hi)` splits into
  32 MiB chunks across 8 threads.
- `rg_cache/` — on-disk row-group cache, evicted by `_enforce_rg_cache_limit`
  at a hardcoded 12 GiB, and by `_enforce_lru_cache(max_shards=4)`.

Total corpus: 103 shards, 138,987 rows, ~884 GiB. `n_vis` (visual tokens per
image) ranges 16 … 16,653, median 364.

## Facts already established — do not re-derive

- **The row-group size guard counts rows, not bytes.** `EmbStreamDataset`
  asserts `MAX_RG_ROWS = 128` (`stream.py`, the `assert biggest <= ...` lines).
  A row group holds 128 embeddings whose byte size is
  `128 × n_vis × 4096 × 2`, so the same 128-row guard spans 0.4 GiB at
  `n_vis=364` and 16.3 GiB at `n_vis=16,653`. The OOM occurred on a shard
  whose row groups are 1.8 GiB.
- **Whole-row-group reads.** The fetch path calls
  `pq.ParquetFile(...).read_row_group(rgi, columns=["key","n_vis","vis_bytes"])`
  and then iterates only the wanted row indices. For the 8-row smoke test that
  read ~1.8 GiB to serve ~114 MiB — roughly 16x amplification.
- **A Modal-only fast path exists.** When `_in_modal()`, shards are pulled
  whole via `hf_transfer` and read locally instead of by Range.
- The 12 GiB RG cache is a module constant
  (`_enforce_rg_cache_limit(..., max_bytes=12*2**30)`), not configurable.
- The val split is materialized once to `val_cache_<n>.pt` (~10 MB) so probes
  do not re-stream it; the train split has no equivalent.

## What I do NOT know, and what I want established

Treat each as a hypothesis to test or refute, not as a finding to confirm.
There may be causes here I have not identified, and some of what follows may
be wrong or irrelevant.

**H1 — whole-row-group reads are the wrong granularity.** If only a handful
of rows in a 1.8 GiB group are wanted, reading the group is wasteful by an
amplification factor of ~16x. Is a per-row or per-span read possible with
this parquet layout, and what would it cost (extra round trips, more Range
requests, loss of coalescing)?

**H2 — the eviction policy fights the access pattern.** 12 GiB is evicted
while a 1.8 GiB group is still needed. Does a size-aware or access-pattern-aware
policy help? Is the cache even worth having at this size?

**H3 — the download dominates wall-clock.** One prefetch took 1172 s for
1.8 GiB (~1.6 MB/s). Training steps in the same era took 1.7-2.8 s each. Is
the GPU idle most of the run, and by how much? **This is the measurement I
most want**: the ratio of time spent in `load_span` to time spent in
`train_step_qwen`, on a real run.

**H4 — the plan is drawn without regard to row-group locality.** `build_epoch_plan`
takes whole shards greedily under an `n_vis` bucket budget. Consecutive batches
may therefore jump between shards, so each row group serves very few batches
and the cache never pays off. Is that what happens?

**H5 — something else entirely.** Memory held by the OOM-killed process may
not be only the row group: `bytearray(vb)` plus a numpy view plus the float32
tensor plus the pinned collate buffer could multiply the peak. Or the prefetch
thread and the main thread can hold two groups at once.

## Method

1. **Measure before concluding.** The download/compute ratio (H3) is one number
   that reframes everything else. Get it from a real run's telemetry
   (`step_ms` against the `load_span` timings in the same log) or from a
   small instrumented run.
2. **Do not trust the 12 GiB cap as a design constant.** It was chosen for a
   Colab disk; the target platform is a rented GPU instance with far more
   storage. Say what the right number is for that platform, and what the code
   should do when the disk is small.
3. **Distinguish correctness from throughput.** If a proposed change would
   alter which rows are served or their order, call it out — `start_pos`,
   resume, and the `manifest_sha256` recorded in checkpoints all depend on the
   plan being stable.
4. **Say what you could not verify.** No GPU, no HF network, no 1 TB corpus in
   this sandbox are all constraints; state what each one leaves open.

## Deliverable

- The measured download/compute ratio, with the run it came from.
- A ranked list of throughput problems, each with: what, where
  (`file.py:line`), the amplification or cost it implies (measured, not
  estimated), and whether fixing it risks correctness or resume.
- The smallest change that would give the largest win, and what it costs.
- Anything you believe is already efficient, so nobody spends time on it.
- Explicitly: which of H1-H5 you confirmed, which you refuted, and which you
  could not test.

Do not implement anything. Report only — this is a second opinion before a
refactor that touches the resume path and the training loop's I/O behaviour.