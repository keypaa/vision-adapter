# AGENTS.md — Vision-Adapter

Preferences and conventions for working in this repo. Follow these unless a task
explicitly says otherwise.

## Environment

- Use the local venv: `source .venv/bin/activate`.
- Model/config/code assets live on HuggingFace, not on disk:
  - `keypa/MoonViT-V2-Standalone` (weights, `vision_config.json`, `moonvit.py`, `preprocess.py`)
  - `keypa/vision-adapter-images`, `keypa/vision-adapter-embeddings`, `keypa/vision-adapter-manifests` (datasets)
- **NEVER rely on local copies of `moonvit.py`/`preprocess.py`** — code is fetched from the HF repo at runtime in Modal (`precompute`). Verify against the repo, not the working tree.

## Tests / checks

- Run `python -m pytest test_preprocess.py` (preprocess contracts; must stay green).
- After any edit: `python -c "import ast; ast.parse(open('<file>').read())"`.

## Modal

- App: `vision-adapter`, Volume `vision-adapter-data` mounted at `/data` (read-only unless reloaded via `vol.reload()`), HF cache volume `vision-adapter-hf` at `/hf`.
- Run a function: `modal run modal_pipeline.py::<func> --arg=val`.
- **No `--param` wrapper in the installed Modal version.** CLI args appear directly (`--workers=16`).
- `ANY`-typed CLI params (e.g. tuples) arrive as **strings** — coerce them in the function (see `precompute_bench`).
- Every `print` in a Modal function must use `flush=True` — buffered pipes otherwise hide progress.

## Long-running jobs — mandatory logging

Never let a long-running task run silently. The `precompute`-style contract:
- Timed phase lines at startup: `[precompute] +<s>  <phase>` covering code-download, model load, corpus scan, cache-check, packing.
- Progress heartbeats: every **~100 images or 30 s**, whichever first.
- Progress line must include: `done/total (%),  img/s,  ETA min,  gpu=<alloc>/<peak> GiB`.
- File-scanning loops (137k+ files) emit progress every ~20k files.
- `torch.cuda.empty_cache()` between distinct phases; never accrue 10s-of-GiB of reserved-but-unallocated memory across runs.

## Numbers / corpus

- Full image corpus **≈139k** (79,659 agentic + ~59,328 cauldron). `~145k` in older docstrings is stale — the live count from `glob`/`precompute_bench` is authoritative.
- **120k is the training-manifest target** (54k agentic + 54k doc + 12k conv mix) — a *subset* of the corpus, not the total.

## Precompute / batching

- MoonViT-V2 memory scales with **total patches** (`cu_seqlens`), NOT image count. Pack by patch count, never by image count alone.
- `BATCH_PATCHES`-style greedy packer lives in `pack_patched_batches` (`precompute_colab` has its own `pack_batches`).
- CPU precompute is **not viable** (~0.1–0.3 img/s ⇒ 5–20+ days). Use Modal (A100, ~15 img/s, ~3 h) or Colab T4.
- Prefer more `workers`/`ahead` over a larger batch to raise throughput; raise `patch_cap` only to spend memory you actually have (bench: 1M cap ≈ 65% VRAM, util ~95%).

## Naming / key conventions (compatibility-critical)

- Embedding keys hash the **volume-relative logical path** (`agentic/foo.png` → `sha1(rel)[:20].pt` + `.pt`), so Modal, Colab, and local caches agree on the same filename for the same image.
- Input images: `images/{agentic,cauldron}/*`. Output embeddings: `embeddings/<key>.pt` as BF16 flattened `(n, 4096)` tensors — the shape check `_already_done` validates.
- Preprocessing contract (`preprocess.py`): resize ≤ `MAX_PATCHES`/`MAX_SIDE`, pad to 28, patch − 14. Do not change without updating `test_preprocess.py`.

## Workflow style

- Resumable/idempotent by design; skipping already-done work is expected, not a bug.
- Benchmark before committing long/expensive GPU runs (`precompute_bench`).
- Keep changes minimal and focused; don't bundle unrelated refactors.
- Commit messages follow repo style (`fix:`, `feat:`, `perf:` prefixes; short subject line).