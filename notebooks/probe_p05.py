import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    """Cell 1 — setup: paths, credentials, a streaming runner, and the facts of this box.

    Everything the later cells need is defined HERE, and only here, so that
    re-running a later cell never re-runs this one (marimo re-runs a cell's
    DEPENDENTS when it changes; a cell that depends on nothing else stays
    independent).

    The HF token is NOT in this file (this file is committed) — set it in the
    cell below, or export HF_TOKEN before opening the notebook.
    """
    import os
    import subprocess
    import time
    from pathlib import Path

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
    os.environ["VISION_ADAPTER_CKPT_TAG"] = "p05"

    try:
        import marimo as _mo
    except Exception:  # not running inside marimo -> plain stdout only
        _mo = None

    # The lines that make the cell's OUTPUT AREA read like a live trace.
    # (Everything still goes to the console area; this is the visible part.)
    _NOTABLE = (
        "step ", "VAL ", "plan", "geometry", "slice filter", "val packs",
        "val plan", "PLATEAU", "GROK", "grok", "ERROR", "Error", "error",
        "WARN", "FAIL", "ckpt", "pushed", "resume", "run_end", "samples",
    )

    def emit(line):
        """Show one line in the notebook: console always, output area if notable."""
        print(line, flush=True)
        if _mo is not None:
            try:
                if any(k in line for k in _NOTABLE):
                    _mo.output.append(line)
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

    os.chdir(REPO)
    emit("repo: " + str(REPO))
    emit("HF_TOKEN: " + ("present" if os.environ.get("HF_TOKEN")
                         else "MISSING — the run cannot push checkpoints"))
    emit("run tag: " + os.environ["VISION_ADAPTER_CKPT_TAG"])

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

    disk_facts()

    _head = subprocess.run("git rev-parse --short HEAD", shell=True,
                           capture_output=True, text=True).stdout.strip()
    _up = subprocess.run("git rev-parse --short origin/master", shell=True,
                         capture_output=True, text=True).stdout.strip()
    emit(f"git : HEAD {_head} | origin/master {_up}"
         + ("" if _head == _up else "   <-- PULL FIRST: git pull"))
    return DATA_DIR, REPO, disk_facts, emit, os, run_streaming


@app.cell
def _(run_streaming):
    """Cell 2 — the filtered manifest.

    Drops the 445 HTML-generation and 200 long-answer rows (the source recipe
    says long descriptive answers prevent the grokking event) and KEEPS the
    measured grid_thw. ~1 min, 84 MB.
    """
    run_streaming(
        "python scripts/filter_manifest.py "
        "--manifest data/train_manifest_grids.jsonl "
        "--out data/train_manifest_filtered.jsonl "
        "--max-answer-words 20 --fetch"
    )
    return


@app.cell
def _(run_streaming):
    """Cell 3 — pack the held-out set.

    NOT optional. val_every cannot be set to 0 (the config refuses it), so
    the run probes the held-out set 16 times; without a packed val set each
    probe streams the 28 shards the agentic plan does not use (~260 GiB per
    probe, ~6 h of extra I/O). With packs it is ~3 GiB and local reads.
    """
    run_streaming("python scripts/pack_val.py --n 1000 --out-dir ./data")
    return


@app.cell
def _(os, run_streaming):
    """Cell 4 — THE PROBE (3.5-4.5 h; keep this tab open and watch it).

    agentic slice only (51,809 rows, every row <= 380 visual tokens = the
    reference's 300k-px budget), constant LR 5e-4 + 25-step warmup, measured
    geometry, every question served.

    On-track checkpoints:
      step 250  -> the val probe finishes in < 60 s
      step 500  -> ~25-30 min in (whole-shard path) or ~50 min (Range path)
      end       -> run_end wall_min ~ 200-270
    """
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. The probe pushes its checkpoints and its log to "
            "HF; without a token the run would finish with nothing to show for it. "
            "Set the token in cell 1 and re-run this cell.")
    run_streaming(
        "python -m vision_adapter train --data-dir ./data --config probe "
        "--max-steps 4000 --groups agentic --lr-schedule constant --lr 5e-4 "
        "--warmup-steps 25 --dtype auto "
        "--push-to-hf --hf-ckpt-repo keypa/vision-adapter-probe-checkpoints"
    )
    return


@app.cell
def _(REPO, run_streaming):
    """Cell 5 — read the result against the PRE-REGISTERED table.

    The outcome table is printed from the plan of record
    (docs/DATA_PATH_REBUILD.md 4d) next to the measured numbers, so the
    reading was fixed BEFORE the result was known.
    """
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
