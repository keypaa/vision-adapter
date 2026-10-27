import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Probe #1 — "does the reference recipe grok on our data?"

        This notebook runs **one** experiment and nothing else: the reference
        recipe (constant LR 5e-4), on the slice that is already inside the
        reference's geometry regime (the **agentic** slice — every row ≤ 380
        visual tokens), with the five defects of the old run fixed.

        | | old probe (Sept) | this probe |
        |---|---|---|
        | manifest | plain (synthetic geometry) | **filtered + measured** (`grid_thw`) |
        | slice | all three (73% doc over budget) | **agentic only** (100% ≤380 tokens) |
        | LR | cosine → 0.11× peak | **constant 5e-4** + 25-step warmup |
        | answers | 1,986 rows crushed to ≤2 tokens | **0 crushed** |
        | questions | 61% served | **100% served** |
        | held-out probe | never ran | every 250 steps (16 points) |

        **The order matters — run the cells top to bottom.** Step 1 prepares
        the manifest; Step 2 is the 5-minute gate that catches broken plumbing
        before 4 GPU-hours are spent proving it. Do not skip either step.

        ### Molab rules (they got a previous session banned)

        * **No `nohup`, no `screen`, no SSH.** Run the cells in the foreground.
        * **Keep this tab open and watched.** Use the other cells to look at
          things while it runs — that is what makes the session interactive
          compute rather than a batch job.
        * The run pushes its checkpoints and its log to HF every 100 steps,
          so a killed session is resumable (see the last cell).
        """
    )
    return


@app.cell
def _(mo):
    import os
    import subprocess
    import time
    from pathlib import Path

    import marimo as mo

    REPO = Path("/marimo/vision-adapter")
    DATA_DIR = REPO / "data"

    # --- credentials -------------------------------------------------------
    # Paste your WRITE token here (this cell is never committed), or export
    # HF_TOKEN in the environment before opening the notebook.
    # os.environ["HF_TOKEN"] = "hf_..."
    if os.environ.get("HF_TOKEN", "").strip():
        os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"].strip()
    else:
        os.environ.pop("HF_TOKEN", None)  # anonymous reads are fine; pushes are not

    os.environ["VISION_ADAPTER_PUSH_HF"] = "1"
    os.environ["VISION_ADAPTER_HF_CKPT_REPO"] = "keypa/vision-adapter-probe-checkpoints"
    os.environ["VISION_ADAPTER_CKPT_TAG"] = "p05"   # scopes this run's artifacts

    # The lines that make the cell's OUTPUT AREA read like a live trace.
    _NOTABLE = (
        "step ", "VAL ", "plan", "geometry", "slice filter", "val packs",
        "val plan", "PLATEAU", "GROK", "grok", "ERROR", "Error", "error",
        "WARN", "FAIL", "ckpt", "pushed", "resume", "run_end", "samples",
        "overfit", "PASS",
    )

    def emit(line):
        """Show one line in the notebook: console always, output area if notable."""
        print(line, flush=True)
        try:
            if any(k in line for k in _NOTABLE):
                mo.output.append(line)
        except Exception:
            pass

    def run_streaming(cmd, cwd=None, check=True):
        """Run a shell command and stream its output INTO this notebook cell.

        A child's stdout goes to the process's fd 1, which bypasses marimo's
        capture — so the child's output would land in the terminal instead of
        the cell, and the cell would look frozen for hours. Reading the pipe
        here and re-emitting each line keeps the notebook visibly alive (and
        makes the session look like interactive compute, which is what Molab
        asks for).
        """
        emit(f"$ {cmd}")
        t0 = time.time()
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(cwd or REPO),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            emit(line.rstrip())
        rc = proc.wait()
        emit(f"[{'ok' if rc == 0 else 'FAILED'}] {time.time() - t0:.0f}s  rc={rc}")
        if check and rc != 0:
            raise RuntimeError(f"command failed (rc={rc}): {cmd}")
        return rc

    def disk_facts(paths=(".", "/tmp", "/dev/shm")):
        """The REAL free space, and the write speed of the mount the cache lives on.

        `df` misreports some mounts (a FUSE/overlay mount can report a size of
        8.0E), and the whole-shard cache budget is derived from the free space
        (half of it). So the number that decides the transport is statvfs's,
        not df's — and if the mount lies about its size, the download can fail
        with ENOSPC mid-shard (the loader then falls back to Range: correct,
        but ~1 h slower).
        """
        for p in paths:
            try:
                st = os.statvfs(p)
                free = st.f_bavail * st.f_frsize / 2**30
                total = st.f_blocks * st.f_frsize / 2**30
                emit(f"disk {p:32s} free {free:9.1f} GiB   total {total:9.1f} GiB")
            except Exception as e:  # noqa: BLE001
                emit(f"disk {p:32s} ERR {e}")
        try:
            _p = REPO / "data" / ".wtest"
            _p.parent.mkdir(parents=True, exist_ok=True)
            _t = time.time()
            with open(_p, "wb") as f:
                f.write(os.urandom(256 * 2**20))
                f.flush()
                os.fsync(f.fileno())
            _dt = max(time.time() - _t, 1e-9)
            _p.unlink(missing_ok=True)
            emit(f"write 256 MiB to the data dir: {_dt:.1f}s -> {256/_dt:.0f} MiB/s")
        except Exception as e:  # noqa: BLE001
            emit(f"write probe failed: {e}")

    os.chdir(REPO)
    emit("repo: " + str(REPO))
    emit("HF_TOKEN: " + ("present" if os.environ.get("HF_TOKEN")
                         else "MISSING — the run cannot push checkpoints"))
    emit("run tag: " + os.environ["VISION_ADAPTER_CKPT_TAG"])
    disk_facts()
    _head = subprocess.run("git rev-parse --short HEAD", shell=True,
                           capture_output=True, text=True).stdout.strip()
    _up = subprocess.run("git rev-parse --short origin/master", shell=True,
                         capture_output=True, text=True).stdout.strip()
    emit(f"git : HEAD {_head} | origin/master {_up}"
         + ("" if _head == _up else "   <-- PULL FIRST (step 0)"))
    return DATA_DIR, Path, REPO, disk_facts, emit, mo, os, run_streaming, subprocess, time


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What a fresh session has, and what it does not

        Every session starts empty, so it is worth being explicit about what
        survives and what has to be re-fetched:

        | | survives a fresh session? | cost to rebuild |
        |---|---|---|
        | the repo | **yes** (it is a clone you pull) | seconds |
        | the run's checkpoints + log | **yes** — on HF, pushed every 100 steps | seconds to download |
        | the filtered manifest | no | 84 MB (~1 min) — step 1 does it with `--fetch` |
        | the key index | no | 28 s, automatic |
        | the held-out packs | no | **published now** → ~3 GB (seconds); a build is ~175 GB (~15–25 min) |
        | the corpus shards | no | **246 GiB** for the agentic slice — unavoidable, ~25 min at 166 MiB/s |
        | the shard cache | no | it is the disk's job, not the network's |

        So a fresh session costs roughly **30–45 minutes of downloads** before
        the first training step, then 3.5–5 hours of run. That is the price
        of not keeping a volume around — and it is why the *pack* is
        published: it is the one artefact that is cheap to keep and
        expensive to rebuild.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 0 — sync the code, and clear the old temp shards

        Two things, both seconds:

        1. **`git pull`** — the code has moved since the last attempt (the
           held-out pack's row-index fix, the pack's cache fix, the overfit
           test).
        2. **`rm -rf /tmp/valpack_*`** — the previous pack step downloaded
           whole shards into a temp dir it then threw away. On a box with a
           small `/tmp` (often RAM-backed) that fills the mount and the
           downloads start failing. The pack step now uses the trainer's own
           cache (`data/cache`) instead, so nothing goes to `/tmp` any more.

        **Then look at the disk line above.** The shard cache budget is *half
        the free space*, so the free space decides the transport:

        | free space on `data/` | what the loader does | total wall clock |
        |---|---|---|
        | ≥ 300 GiB | whole-shard path: 246 GiB fetched once | **~3.5 h** |
        | 60–300 GiB | partial: some shards re-fetched | ~4 h |
        | < 60 GiB | Range path (~3× overfetch) | ~4.5–6 h |

        All three work. The slow ones are just slow.
        """
    )
    return


@app.cell
def _(REPO, emit, run_streaming):
    run_streaming("git pull")
    run_streaming("rm -rf /tmp/valpack_*")   # the old temp shards (may not exist)
    run_streaming("ls -la data/ | head -20")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 1 — the filtered manifest (measured geometry, no long answers)

        Run this before the 5-minute gate: the gate needs a local train
        manifest, and `--fetch` downloads the measured source manifest when
        this is a fresh session.

        Drops the **645** rows the source recipe says prevent the grokking
        event (445 HTML-generation targets + 200 rationale-style answers) and
        **keeps the measured `grid_thw`** on every row.

        Expected output (it is a check, not a surprise):

        ```
        [filter] 117600 -> 116955 rows (dropped 645: {'html_target': 445, 'long_answer': 200})
        [filter] kept by group: {'agentic': 52899, 'doc': 52453, 'conv': 11603}
        ```

        The agentic slice is **52,899 rows** in the manifest, of which
        **51,809 are servable** (the rest have no embedding) — and 100% of
        them are ≤ 380 visual tokens, i.e. inside the reference's 300k-pixel
        budget. That is the whole point of this probe.
        """
    )
    return


@app.cell
def _(run_streaming):
    run_streaming(
        "python scripts/filter_manifest.py "
        "--manifest data/train_manifest_grids.jsonl "
        "--out data/train_manifest_filtered.jsonl "
        "--max-answer-words 20 --fetch"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 2 — the 5-minute gate: can the projector even move?

        This is the cheapest experiment in the project and it decides whether
        the 4-hour probe is worth running.

        **The question:** train on **16 fixed rows** and see whether the loss
        falls. Memorising 16 rows is trivial for a projector. If the loss
        will not move, then the collate, the position ids, the LR or the
        projector scale are broken — and a 4,000-step probe would spend four
        hours proving exactly that.

        It uses the **same** manifest, key index, plan, dataset, collate,
        projector init, optimizer and forward step as the probe, so this is
        evidence about the probe, not about a toy.

        **How to read it:**

        | output | meaning | what to do |
        |---|---|---|
        | `-> PASS`, loss 1.1 → <0.2 | the plumbing works | **continue to step 3** |
        | `-> FAIL`, loss stays ≈1.1 | a bug, not a science problem | **stop** — paste the output here, do not run the probe |
        | non-finite loss | the LR/scale is off | stop, same as FAIL |

        Cost: ~1 min model load + 40 steps × ~2 s ≈ **3–5 minutes**, and it
        downloads only 16 rows' embeddings (~50 MB).
        """
    )
    return


@app.cell
def _(run_streaming):
    run_streaming(
        "python scripts/overfit_test.py --data-dir ./data --rows 16 --steps 40 "
        "--lr 5e-4 --groups agentic"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 3 — pack the held-out set (not optional)

        The trainer probes a held-out set every 250 steps (16 times). The
        held-out rows live in the corpus shards; without a local pack, each
        probe streams the shards the training plan does not use — ~260 GiB
        per probe, **~6 hours of extra I/O** for the run.

        With the pack it is ~3 GiB and every probe is a local read.

        **A fresh session has no `data/` and no caches**, so this step either
        *downloads the published packs* (~3 GB, seconds) or *builds them*
        (~50 shards, ~175 GB of transfer, ~15-25 min) and publishes them for
        next time. The cell below tries the published ones first.

        **Two things to know about a build:**

        * The shards are downloaded **whole** (pyarrow needs a local parquet
          file), ~50 of them ≈ 175 GB of transfer. They land in the
          trainer's own cache (`data/cache`) and the LRU trims it, so the disk
          peak stays inside the budget — **not** in `/tmp`, which is small.
        * Expect a few minutes of downloading. Watch for one line per shard.

        **Expected output:** ~629 agentic rows in the pack (the val split holds
        1,076 agentic rows; the quota comes from the corpus mix). If the pack
        says far fewer, stop and paste the output — the trainer's coverage
        check would then fall back to the slow path and the run would take 6 h
        longer than you want.
        """
    )
    return


@app.cell
def _(emit, run_streaming):
    _have_packs = False
    try:
        from huggingface_hub import hf_hub_download

        for _g in ("agentic", "doc", "conv"):
            for _f in (f"val_pack_{_g}.parquet", f"val_pack_{_g}.map.json"):
                hf_hub_download("keypa/vision-adapter-manifests", _f,
                                repo_type="dataset", local_dir="./data")
        _have_packs = True
        emit("[notebook] published held-out packs downloaded -> ./data")
    except Exception as _e:  # noqa: BLE001 — not published yet: build them
        emit(f"[notebook] no published packs ({type(_e).__name__}) — building them")
    if not _have_packs:
        run_streaming("python scripts/pack_val.py --n 1000 --out-dir ./data --publish")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 4 — the probe (3.5–5 h; keep this tab open and watch it)

        ```
        python -m vision_adapter train --data-dir ./data --config probe \\
            --max-steps 4000 --groups agentic --lr-schedule constant --lr 5e-4 \\
            --warmup-steps 25 --dtype auto \\
            --push-to-hf --hf-ckpt-repo keypa/vision-adapter-probe-checkpoints
        ```

        **What it is:** 4,000 steps × batch 16 = **64,000 samples** on the
        51,809-row agentic slice = **1.23 epochs**. The reference collapsed
        at 57,600 samples (87% of its epoch) → the event should land at step
        **~2,810** (epoch-relative) or **~3,600** (sample-relative). The run
        covers both.

        **The first two minutes — check these three lines:**

        | line | must say |
        |---|---|
        | `[train] geometry: …` | `grid_source=measured`, and **no** "carries no grid_thw" warning |
        | `[train] slice filter:` | `-> 52899/116955 rows` |
        | `[train] val packs:` | `629 rows in 1 local file ({'agentic': 629})` |

        **On-track checkpoints while it runs:**

        | when | expect | if not |
        |---|---|---|
        | ~10 min | step 250, and its val probe finished in **< 60 s** | minutes → the packs weren't picked up → stop |
        | ~25–30 min | step 500 | ~50 min → Range path (fine); ~90 min → stop |
        | end | `run_end … wall_min ≈ 200–300` | — |

        **Artifacts** (tagged `p05`, so the old run's files in the HF repo are
        untouched): `projector_p05_step*.pt`, `probe_log_p05.jsonl`,
        `probe_curves_p05.png`.
        """
    )
    return


@app.cell
def _(os, run_streaming):
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. The probe pushes its checkpoints and its log to "
            "HF; without a token the run would finish with nothing to show for it. "
            "Set the token in the setup cell and re-run this one.")
    run_streaming(
        "python -m vision_adapter train --data-dir ./data --config probe "
        "--max-steps 4000 --groups agentic --lr-schedule constant --lr 5e-4 "
        "--warmup-steps 25 --dtype auto "
        "--push-to-hf --hf-ckpt-repo keypa/vision-adapter-probe-checkpoints"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## While it runs — and if it dies

        **Look at things.** In another cell:

        ```python
        import subprocess
        print(subprocess.run("tail -5 /marimo/vision-adapter/data/probe_log_p05.jsonl",
                             shell=True, capture_output=True, text=True).stdout)
        ```

        **If the session dies** (Molab terminates notebooks used for
        non-interactive jobs, and an idle session can time out), the run is
        resumable: checkpoints are pushed to HF every 100 steps. Re-run the
        probe cell with `--resume hf` added — and keep the `p05` tag in the
        environment, or the resume will pick the *old* run's checkpoint
        (`projector_step4000.pt`) instead of this one's.

        ```python
        run_streaming(
            "python -m vision_adapter train --data-dir ./data --config probe "
            "--max-steps 4000 --groups agentic --lr-schedule constant --lr 5e-4 "
            "--warmup-steps 25 --dtype auto "
            "--push-to-hf --hf-ckpt-repo keypa/vision-adapter-probe-checkpoints "
            "--resume hf"
        )
        ```

        **Do not** set `VISION_ADAPTER_ALLOW_LEGACY_RESUME=1`: that flag exists
        to continue an *old-format* checkpoint onto a different row stream,
        which would silently train on different data.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## After the run — read the result against the PRE-REGISTERED table

        The outcome table was written **before** the run (so it cannot be
        fitted to the result). The cell below prints the measured analysis
        next to it.

        The grok criterion is fixed in advance: a **sustained ≥0.3-nat drop in
        the EMA *level***, holding for ≥3 logged windows, at or after step
        ~2,800. (The reference's own ">0.5 nats below the running mean" test
        fires **271 times** on a per-step log like ours — it measures noise
        here, so it is not used.)

        The three numbers that decide the next step:

        1. the **sustained grok criterion** (above),
        2. **`g_mix` per step** — which slice learns,
        3. **`Σ wait_ms / Σ step_ms`** — the data path's share of the wall.

        The val loss is a *secondary* signal: the val split carries no
        measured geometry, so it runs on the synthetic grid — its number is an
        inflated upper bound and only its **movement** is meaningful.
        """
    )
    return


@app.cell
def _(REPO, emit, run_streaming):
    _log = REPO / "data" / "probe_log_p05.jsonl"
    if not _log.is_file():
        # A fresh session has no data/ — the log of a run from an EARLIER
        # session lives on HF (pushed every save). Fetch it so the reading
        # can happen in any session.
        emit("[notebook] no local log — fetching it from HF")
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download("keypa/vision-adapter-probe-checkpoints",
                            "probe_log_p05.jsonl", repo_type="model",
                            local_dir=str(REPO / "data"))
        except Exception as _e:  # noqa: BLE001
            emit(f"[notebook] no log on HF either ({type(_e).__name__}) — "
                 "nothing to analyse yet")
    run_streaming(
        "python scripts/analyze_probe.py --log data/probe_log_p05.jsonl "
        "--out probe_p05.md"
    )
    _report = REPO / "probe_p05.md"
    if _report.is_file():
        print("\n" + "=" * 70 + "\n" + _report.read_text()[:6000])
    _doc = REPO / "docs" / "DATA_PATH_REBUILD.md"
    if _doc.is_file():
        _text = _doc.read_text()
        _i = _text.find("## 4d. PRE-REGISTERED OUTCOMES")
        _j = _text.find("\n---", _i)
        if _i > 0:
            print("\n" + "=" * 70
                  + "\nPRE-REGISTERED OUTCOMES (docs/DATA_PATH_REBUILD.md 4d)\n"
                  + _text[_i:_j if _j > _i else _i + 6000][:6000])
    return


if __name__ == "__main__":
    app.run()
