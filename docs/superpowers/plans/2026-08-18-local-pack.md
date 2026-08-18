# Local Embedding Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone, resumable, RAM-bounded local script (`local_pack.py`) that converts the ~139k `.pt` MoonViT-V2 embeddings on the Modal Volume into ~100 × 10 GB parquet shards, uploads each shard to the Volume (`/data/shards/`), and pushes it to HF (`keypa/vision-adapter-embeddings`), running overnight on a 6-core / 16 GB laptop over 1 Gb/s fiber.

**Architecture:** One focused file, `local_pack.py`, whose functions take their I/O dependencies (`vol`, `HfApi`) as parameters so they're unit-testable with fakes. Pure logic (shard naming, row construction, streaming pack, resume decisions) is fully tested with `pytest`; thin I/O wrappers (download/upload/push) are tested with a small fake volume and fake HF API, and wired together in an integration test over a tiny fake corpus on local disk.

**Tech Stack:** `modal` (volume I/O), `torch` (>=2.0 `weights_only=True`), `pyarrow`, `numpy`, `huggingface_hub`.

## Global Constraints

- `shard_rows = 1360` (must stay 1360 for byte-compatible shard boundaries vs the existing smoke shards `emb_0000`/`emb_0001` and vs Phase 2's spec).
- RAM cap 16 GB → streaming `ParquetWriter`; never hold a full shard's rows — batch of 64 rows written per `write_table` call.
- CPU/threads: download pool and any parallelism capped at 6 (laptop core count); `torch.load` not parallelized.
- Volume mount root paths: `embeddings/<sha1>.pt` (source), `shards/emb_{i:04d}.parquet` (output).
- HF layout: `data/emb_{i:04d}.parquet` in repo `keypa/vision-adapter-embeddings` (`repo_type="dataset"`).
- Schema must match the Modal pack exactly: `key` (string, value `embeddings/<sha1>.pt`), `n_vis` (int64), `vis_bytes` (binary).
- `vis_bytes` must equal `tensor.view(torch.uint8).numpy().tobytes()` on the original 2-D bf16 tensor (shape `[n_merged, 4096]`), no compression.
- Resumable semantics per shard: skip if on volume **and** HF; `push_from_vol` if only on volume (pull parquet → push HF, no re-pack); else `pack` (download `.pt` → pack → upload volume → push HF). Local `.pt`/parquet staging files are deleted on success.
- Verification gate per shard: row count == slice length; round-trip `frombuffer→reshape(-1,4096)` == original tensor for a sample row.
- Tests live at repo root as `test_local_pack.py` (alongside `test_preprocess.py`); run with `python -m pytest test_local_pack.py -q` from the venv at repo root.
- After every edit: `python -c "import ast; ast.parse(open('local_pack.py').read())"`.

---

### Task 1: Skeleton, constants, and pure shard helpers

**Files:**
- Create: `local_pack.py`
- Create: `test_local_pack.py`

**Interfaces produced:**
- `sorted_embedding_names(entries: list) -> list[str]` (pure)
- `shard_slices(names: list[str], shard_rows: int) -> list[list[str]]` (pure)

- [ ] **Step 1: Write failing tests**

```python
# test_local_pack.py
from local_pack import sorted_embedding_names, shard_slices, FileEntryLike


def test_sorted_embedding_names_is_sorted_and_strips_prefix():
    entries = [
        FileEntryLike(path="embeddings/zzz.pt"),
        FileEntryLike(path="embeddings/aaa.pt"),
        FileEntryLike(path="embeddings/mmm.pt"),
    ]
    assert sorted_embedding_names(entries) == [
        "embeddings/aaa.pt",
        "embeddings/mmm.pt",
        "embeddings/zzz.pt",
    ]


def test_shard_slices_boundaries():
    names = [f"embeddings/{i:04d}.pt" for i in range(7)]
    s = shard_slices(names, 3)
    assert s == [
        ["embeddings/0000.pt", "embeddings/0001.pt", "embeddings/0002.pt"],
        ["embeddings/0003.pt", "embeddings/0004.pt", "embeddings/0005.pt"],
        ["embeddings/0006.pt"],
    ]
```

- [ ] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py::test_sorted_embedding_names_is_sorted_and_strips_prefix test_local_pack.py::test_shard_slices_boundaries -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# local_pack.py
import os

SHARD_ROWS = 1360
VOL_NAME = "vision-adapter-data"
EMB_REPO = "keypa/vision-adapter-embeddings"
REPO_TYPE = "dataset"


class FileEntryLike:
    """Minimal FileEntry shim for tests (only `path` is needed)."""

    __slots__ = ("path",)

    def __init__(self, path):
        self.path = path


def sorted_embedding_names(entries):
    """Return sorted `embeddings/<sha1>.pt` paths (matches Modal's sorted(glob))."""
    return sorted(e.path for e in entries if e.path.startswith("embeddings/"))


def shard_slices(names, shard_rows):
    """Contiguous slices of `names` aligned to Modal's shard numbering."""
    out = []
    for i in range(0, len(names), shard_rows):
        out.append(names[i : i + shard_rows])
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add local_pack.py test_local_pack.py
git commit -m "feat: add pure shard enumeration + slicing helpers"
```

---

### Task 2: Row construction, streaming pack, and round-trip

**Files:**
- Modify: `local_pack.py`
- Modify: `test_local_pack.py`

**Interfaces produced:**
- `make_row(path: str, tensor: torch.Tensor) -> dict`
- `iter_rows(local_paths: list[str]) -> Iterator[dict]`
- `pack_rows(rows: Iterable[dict], out_path: str, batch_size: int = 64) -> None`

**Interfaces consumed:** `sorted_embedding_names` (none here), torch, pyarrow.

- [ ] **Step 1: Write failing tests**

```python
import io
import numpy as np
import pyarrow.parquet as pq
import torch
from local_pack import make_row, iter_rows, pack_rows


def _bf16(rows, cols=4096):
    return torch.randn(rows, cols, dtype=torch.bfloat16)


def test_make_row_shape_and_key():
    t = _bf16(5)
    r = make_row("embeddings/abc.pt", t)
    assert r["key"] == "embeddings/abc.pt"
    assert r["n_vis"] == 5
    # bf16 is 2 bytes -> 5*4096*2 = 40960 bytes
    assert len(r["vis_bytes"]) == 5 * 4096 * 2


def test_pack_rows_schema_roundtrip(tmp_path):
    rows = [
        make_row("embeddings/a.pt", _bf16(3)),
        make_row("embeddings/b.pt", _bf16(7)),
        make_row("embeddings/c.pt", _bf16(2)),
    ]
    out = tmp_path / "emb_0000.parquet"
    pack_rows(rows, str(out), batch_size=2)
    tbl = pq.read_table(str(out))
    assert tbl.num_rows == 3
    assert tbl.column_names == ["key", "n_vis", "vis_bytes"]
    first = tbl.slice(0, 1).to_pylist()[0]
    assert first["key"] == "embeddings/a.pt"
    assert first["n_vis"] == 3
    rt = (
        torch.from_numpy(np.frombuffer(first["vis_bytes"], dtype=np.uint8))
        .view(torch.bfloat16)
        .reshape(-1, 4096)
    )
    assert torch.equal(rt, _bf16_unused())  # placeholder, replaced below
```

Replace the placeholder assertion with a held-out original tensor:

```python
def test_pack_rows_schema_roundtrip(tmp_path):
    t0, t1, t2 = _bf16(3), _bf16(7), _bf16(2)
    rows = [
        make_row("embeddings/a.pt", t0),
        make_row("embeddings/b.pt", t1),
        make_row("embeddings/c.pt", t2),
    ]
    out = tmp_path / "emb_0000.parquet"
    pack_rows(rows, str(out), batch_size=2)
    tbl = pq.read_table(str(out))
    assert tbl.num_rows == 3
    assert tbl.column_names == ["key", "n_vis", "vis_bytes"]
    row_b = tbl.slice(1, 1).to_pylist()[0]
    assert row_b["key"] == "embeddings/b.pt"
    assert row_b["n_vis"] == 7
    rt = (
        torch.from_numpy(np.frombuffer(row_b["vis_bytes"], dtype=np.uint8))
        .view(torch.bfloat16)
        .reshape(-1, 4096)
    )
    assert torch.equal(rt, t1)  # round-trip equality == original tensor


def test_iter_rows_roundtrip(tmp_path):
    t = _bf16(4)
    p = tmp_path / "xyz.pt"
    torch.save(t, str(p))
    r = make_row("embeddings/xyz.pt", t)
    packed = list(iter_rows([str(p)]))
    assert packed[0]["key"] == "embeddings/xyz.pt"
    assert packed[0]["n_vis"] == 4
    rt = (
        torch.from_numpy(np.frombuffer(packed[0]["vis_bytes"], dtype=np.uint8))
        .view(torch.bfloat16)
        .reshape(-1, 4096)
    )
    assert torch.equal(rt, t)
```

Add imports at the top of the test file: `from local_pack import iter_rows` (and `torch`, `numpy`).

- [ ] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py::test_make_row_shape_and_key test_local_pack.py::test_pack_rows_schema_roundtrip test_local_pack.py::test_iter_rows_roundtrip -v
```

Expected: FAIL (functions missing / `import torch` not at top of module).

- [ ] **Step 3: Write minimal implementation**

```python
# append to local_pack.py (imports already present: add torch, pyarrow, numpy)
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

SCHEMA = pa.schema(
    [
        pa.field("key", pa.string()),
        pa.field("n_vis", pa.int64()),
        pa.field("vis_bytes", pa.binary()),
    ]
)


def make_row(path: str, tensor: torch.Tensor) -> dict:
    assert tensor.dim() == 2 and tensor.shape[-1] == 4096, path
    return {
        "key": path,
        "n_vis": int(tensor.shape[0]),
        "vis_bytes": tensor.view(torch.uint8).numpy().tobytes(),
    }


def iter_rows(local_paths):
    """torch.load each staged .pt, yield a row dict (key = embeddings/<basename>)."""
    for p in local_paths:
        t = torch.load(p, map_location="cpu", weights_only=True)
        yield make_row(f"embeddings/{os.path.basename(p)}", t)


def pack_rows(rows, out_path, batch_size=64):
    """Stream rows to a parquet file in fixed-size batches (RAM-bounded)."""
    writer = pq.ParquetWriter(out_path, SCHEMA)
    batch = []
    for r in rows:
        batch.append(r)
        if len(batch) >= batch_size:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            batch.clear()
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    writer.close()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add local_pack.py test_local_pack.py
git commit -m "feat: add streaming pack + round-trip row construction"
```

---

### Task 3: Resume decision logic + state probes

**Files:**
- Modify: `local_pack.py`
- Modify: `test_local_pack.py`

**Interfaces produced:**
- `existing_volume_shards(vol) -> set[str]`
- `existing_hf_shards(api, repo_id: str) -> set[str]`
- `resume_action(shard: str, vol_shards: set, hf_shards: set) -> str` (returns `skip` | `push_from_vol` | `pack`)

**Interfaces consumed:** `shard_slices` (naming).

- [ ] **Step 1: Write failing tests**

```python
from local_pack import resume_action, existing_volume_shards, existing_hf_shards


def test_resume_action_skip_pack_push_from_vol():
    assert resume_action("emb_0002.parquet", {"emb_0002.parquet"}, {"emb_0002.parquet"}) == "skip"
    assert resume_action("emb_0002.parquet", set(), set()) == "pack"
    assert resume_action("emb_0002.parquet", {"emb_0002.parquet"}, set()) == "push_from_vol"
    # HF-only is treated as recoverable (volume copy missing) -> pack fresh
    assert resume_action("emb_0002.parquet", set(), {"emb_0002.parquet"}) == "pack"


def _fakedisk():
    class FakeVol:
        def __init__(self, vol_paths):
            self._paths = vol_paths
        def listdir(self, path):
            prefix = path.rstrip("/") + "/"
            return [FileEntryLike(p) for p in self._paths if p.startswith(prefix)]
    return FakeVol


def _fakeapi(files):
    class FakeApi:
        def list_repo_files(self, repo_id, repo_type=None):
            return files
    return FakeApi


def test_existing_volume_shards_strips_prefix(tmp_path):
    vol = _fakedisk()(["shards/emb_0000.parquet", "shards/emb_0001.parquet", "shards/.keep"])
    assert existing_volume_shards(vol) == {"emb_0000.parquet", "emb_0001.parquet"}


def test_existing_hf_shards_strips_prefix():
    api = _fakeapi(["data/emb_0000.parquet", "data/emb_0001.parquet", "dataset_infos.json", ".gitattributes"])
    assert existing_hf_shards(api, "keypa/vision-adapter-embeddings") == {
        "emb_0000.parquet", "emb_0001.parquet",
    }
```

- [ ] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py::test_resume_action_skip_pack_push_from_vol test_local_pack.py::test_existing_volume_shards_strips_prefix test_local_pack.py::test_existing_hf_shards_strips_prefix -v
```

Expected: FAIL (missing functions).

- [ ] **Step 3: Write minimal implementation**

```python
import os


def existing_volume_shards(vol):
    out = set()
    try:
        for e in vol.listdir("shards"):
            if e.path.endswith(".parquet"):
                out.add(os.path.basename(e.path))
    except Exception:
        pass  # no shards dir yet
    return out


def existing_hf_shards(api, repo_id):
    out = set()
    try:
        files = api.list_repo_files(repo_id, repo_type=REPO_TYPE)
    except Exception:
        return out
    for n in files:
        base = os.path.basename(n)
        if base.startswith("emb_") and base.endswith(".parquet"):
            out.add(base)
    return out


def resume_action(shard, vol_shards, hf_shards):
    on_vol = shard in vol_shards
    on_hf = shard in hf_shards
    if on_vol and on_hf:
        return "skip"
    if on_vol and not on_hf:
        return "push_from_vol"
    return "pack"
```

- [ ] **Step 4: Run tests to verify pass**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py -q
```

Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add local_pack.py test_local_pack.py
git commit -m "feat: add resume-state probes and shard action decision"
```

---

### Task 4: Volume I/O: download stage, upload, and HF push

**Files:**
- Modify: `local_pack.py`
- Modify: `test_local_pack.py`

**Interfaces produced:**
- `download_shard(vol, shard_names: list[str], stage_dir: str, workers: int = 6, retries: int = 3) -> list[str]`
- `upload_to_volume(vol, local_path: str, shard: str) -> None`
- `push_to_hf(api, local_path: str, repo_id: str, shard: str) -> None`
- `pull_volume_parquet(vol, shard: str, dst: str, retries: int = 3) -> None`  (used by `push_from_vol`)

**Dependencies:** `concurrent.futures`, `time`, `huggingface_hub.HfApi` (lazy import), `modal.Volume`.

- [ ] **Step 1: Write failing tests**

```python
import torch as _torch  # keep test-local alias
from local_pack import download_shard, upload_to_volume, push_to_hf, pull_volume_parquet


class FakeVol:
    """In-memory volume: read_file_into_fileobj + batch_upload + listdir."""

    def __init__(self, files=None):
        self.files = files or {}          # path -> bytes
        self.uploaded = {}               # shard -> local_path
        self.committed = 0

    def read_file_into_fileobj(self, path, fileobj, progress_cb=None):
        fileobj.write(self.files[path])
        return len(self.files[path])

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        return [FileEntryLike(p) for p in self.files if p.startswith(prefix)]

    class _Batch:
        def __init__(self, outer): self.outer = outer
        def put_file(self, local, remote): self.outer.uploaded[os.path.basename(remote)] = local
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def batch_upload(self, force=False):
        return FakeVol._Batch(self)

    def commit(self):
        self.committed += 1


class FakeApi:
    def __init__(self):
        self.calls = []
    def list_repo_files(self, repo_id, repo_type=None):
        return []
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
        self.calls.append((path_or_fileobj, path_in_repo, repo_id))
        return "ok"


def _make_pt(path, tensor):
    _torch.save(tensor, path)
    return open(path, "rb").read()


def test_download_shard_writes_files(tmp_path):
    vol = FakeVol()
    buf = io.BytesIO(); _torch.save(_bf16(3), buf); data = buf.getvalue()
    vol.files["embeddings/a.pt"] = data
    vol.files["embeddings/b.pt"] = data
    out = download_shard(vol, ["embeddings/a.pt", "embeddings/b.pt"], str(tmp_path / "stage"), workers=1)
    assert len(out) == 2
    assert (tmp_path / "stage" / "a.pt").read_bytes() == data


def test_upload_to_volume_records_shard(tmp_path):
    vol = FakeVol()
    p = tmp_path / "emb_0000.parquet"
    p.write_bytes(b"parquet-bytes")
    upload_to_volume(vol, str(p), "emb_0000.parquet")
    assert vol.uploaded["emb_0000.parquet"] == str(p)


def test_push_to_hf_invokes_api(tmp_path):
    api = FakeApi()
    p = tmp_path / "emb_0000.parquet"
    p.write_bytes(b"x")
    push_to_hf(api, str(p), "keypa/vision-adapter-embeddings", "emb_0000.parquet")
    assert api.calls == [(str(p), "data/emb_0000.parquet", "keypa/vision-adapter-embeddings")]


def test_pull_volume_parquet_writes_local(tmp_path):
    vol = FakeVol()
    vol.files["shards/emb_0099.parquet"] = b"parquet-bytes"
    dst = tmp_path / "pulled.parquet"
    pull_volume_parquet(vol, "emb_0099.parquet", str(dst))
    assert dst.read_bytes() == b"parquet-bytes"
```

Add `import io` if not already at top of test file.

- [ ] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py::test_download_shard_writes_files test_local_pack.py::test_upload_to_volume_records_shard test_local_pack.py::test_push_to_hf_invokes_api test_local_pack.py::test_pull_volume_parquet_writes_local -v
```

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
import time
from concurrent.futures import ThreadPoolExecutor


def _vol_read_retry(vol, name, dst, retries, delay=0.5):
    last = None
    for attempt in range(retries):
        try:
            with open(dst, "wb") as f:
                vol.read_file_into_fileobj(name, f)
            return dst
        except Exception as e:
            last = e
            time.sleep(delay * (2 ** attempt))
    raise last


def download_shard(vol, shard_names, stage_dir, workers=6, retries=3):
    os.makedirs(stage_dir, exist_ok=True)

    def _one(name):
        dst = os.path.join(stage_dir, os.path.basename(name))
        return _vol_read_retry(vol, name, dst, retries)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, shard_names))


def upload_to_volume(vol, local_path, shard):
    with vol.batch_upload(force=True) as batch:
        batch.put_file(local_path, f"shards/{shard}")
    vol.commit()


def pull_volume_parquet(vol, shard, dst, retries=3):
    _vol_read_retry(vol, f"shards/{shard}", dst, retries=3)


def push_to_hf(api, local_path, repo_id, shard):
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"data/{shard}",
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message=f"Add {shard}",
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add local_pack.py test_local_pack.py
git commit -m "feat: add volume download/upload/push wrappers with retry"
```

---

### Task 5: Orchestration (run_shard + main with heartbeats)

**Files:**
- Modify: `local_pack.py`
- Modify: `test_local_pack.py`

**Interfaces produced:**
- `run_shard(vol, api, i: int, all_names: list[str], shard_rows: int, stage_dir: str, em_repo: str, workers: int = 6, batch_size: int = 64) -> str`
- `main(argv: list[str] | None = None) -> None`

**Interfaces consumed:** all previous.

- [ ] **Step 1: Write a failing integration test**

```python
import shutil
import pyarrow.parquet as pq
from local_pack import run_shard


def _seed_vol(files: dict, tensors: "dict[str, _torch.Tensor]"):
    vol = FakeVol()
    for name, t in tensors.items():
        buf = io.BytesIO(); _torch.save(t, buf)
        vol.files[f"embeddings/{name}"] = buf.getvalue()
    return vol


def test_run_shard_pack_uploads_and_pushes(tmp_path):
    names = [f"embeddings/e{i:04d}.pt" for i in range(4)]
    tensors = {os.path.basename(n): _bf16(3) for n in names}
    vol = _seed_vol({}, tensors)
    api = FakeApi()

    # shard 0 -> indices 0,1 ; shard_rows=2
    action = run_shard(vol, api, 0, names, shard_rows=2,
                       stage_dir=str(tmp_path / "stage"), em_repo="keypa/vision-adapter-embeddings", workers=1)
    assert action in ("pack", "push_from_vol")
    # volume got the parquet
    assert "emb_0000.parquet" in vol.uploaded
    # HF got pushed
    assert any(p == "data/emb_0000.parquet" for _, p, _ in api.calls)


def test_run_shard_skips_when_done(tmp_path):
    names = [f"embeddings/e{i:04d}.pt" for i in range(4)]
    vol = FakeVol()
    vol.files["shards/emb_0000.parquet"] = b"x"
    api = FakeApi()
    # make api claim emb_0000 parquet already present
    api_real = type(api)

    class ApiDone(FakeApi):
        def list_repo_files(self, repo_id, repo_type=None):
            return ["data/emb_0000.parquet"]

    action = run_shard(vol, ApiDone(), 0, names, shard_rows=2,
                       stage_dir=str(tmp_path / "stage"), em_repo="keypa/vision-adapter-embeddings", workers=1)
    assert action == "skip"
```

- [ ] **Step 2: Run to verify failure**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py::test_run_shard_pack_uploads_and_pushes test_local_pack.py::test_run_shard_skips_when_done -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

```python
from typing import Iterable

def _shard_name(i: int) -> str:
    return f"emb_{i:04d}.parquet"


def run_shard(vol, api, i, all_names, shard_rows, stage_dir, em_repo,
              workers=6, batch_size=64, retries=3):
    shard = _shard_name(i)
    chunk = all_names[i * shard_rows : (i + 1) * shard_rows]
    assert chunk, f"shard {i} has no rows"

    action = resume_action(shard, existing_volume_shards(vol), existing_hf_shards(api, em_repo))
    local_parquet = os.path.join(stage_dir, shard)
    os.makedirs(stage_dir, exist_ok=True)

    if action == "skip":
        print(f"[local-pack] shard {i}: {shard} already on volume+HF — skipping", flush=True)
        return action

    if action == "push_from_vol":
        pull_volume_parquet(vol, shard, local_parquet, retries)
    else:
        staged = download_shard(vol, chunk, stage_dir, workers=workers, retries=retries)
        pack_rows(iter_rows(staged), local_parquet, batch_size=batch_size)
        upload_to_volume(vol, local_parquet, shard)
        for p in staged:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    push_to_hf(api, local_parquet, em_repo, shard)
    try:
        os.remove(local_parquet)
    except FileNotFoundError:
        pass
    return action


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-rows", type=int, default=SHARD_ROWS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stage-dir", default="/tmp/emb_stage")
    ap.add_argument("--em-repo", default=EMB_REPO)
    ap.add_argument("--only", default="", help="i[:j] shard range, e.g. 0 or 2:5")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args(argv)

    import modal
    from huggingface_hub import HfApi
    vol = modal.Volume.from_name(VOL_NAME)
    api = HfApi()

    entries = vol.listdir("embeddings")
    names = sorted_embedding_names(entries)
    if not names:
        print("[local-pack] no embeddings found under embeddings/ on the volume — aborting", flush=True)
        return
    slices = shard_slices(names, args.shard_rows)
    print(f"[local-pack] embeddings: {len(names)}  shards: {len(slices)}  "
          f"shard_rows={args.shard_rows}  workers={args.workers}", flush=True)

    lo, hi = 0, len(slices)
    if args.only:
        parts = args.only.split(":")
        lo = int(parts[0]) if parts[0] else 0
        hi = int(parts[1]) if len(parts) > 1 and parts[1] else len(slices)

    t0 = time.time()
    done = 0
    for i in range(lo, hi):
        action = run_shard(vol, api, i, names, args.shard_rows,
                           args.stage_dir, args.em_repo,
                           workers=args.workers, batch_size=args.batch_size,
                           retries=args.retries)
        done += 1
        n = len(slices[i])
        elapsed = time.time() - t0
        rows_done = sum(len(s) for s in slices[lo : lo + done])
        rate = rows_done / max(1e-9, elapsed)
        eta = (len(names) - rows_done) / max(1e-9, rate) / 60
        print(f"[local-pack] done {rows_done}/{len(names)} ({100*rows_done/len(names):.0f}%)  "
              f"{rate:.0f} rows/s  ETA {eta:.0f} min  shard {i}/{hi} ({n} rows) action={action}  "
              f"vol_commits={getattr(vol,'committed',0)}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run full test suite + ast parse**

```bash
source .venv/bin/activate && python -m pytest test_local_pack.py -q
source .venv/bin/activate && python -c "import ast; ast.parse(open('local_pack.py').read())"
```

Expected: all pass, no syntax error.

- [ ] **Step 5: Commit**

```bash
git add local_pack.py test_local_pack.py
git commit -m "feat: add orchestrator + CLI for local embedding pack"
```

---

### Final verification gate (manual, after running against the real volume)

- [ ] `python -c "import ast; ast.parse(open('local_pack.py').read())"` — syntax OK.
- [ ] `source .venv/bin/activate && python -m pytest test_local_pack.py -q` — green.
- [ ] Smoke test one new shard end-to-end against the real volume + HF:
  ```bash
  source .venv/bin/activate
  python local_pack.py --only 2     # packs/skip shard 2 only
  ```
  then inspect `keypa/vision-adapter-embeddings` on HF and `shards/emb_0002.parquet` on the volume; confirm:
  - row count == `SHARD_ROWS` (1360), except possibly the last shard;
  - `sum(emb.n_vis)` for one spot shard equals the number of embeddings in that slice;
  - round-trip: a sample row's `vis_bytes` decodes back to a `[n, 4096]` bf16 tensor that matches the original `.pt`.
- [ ] Full run (after smoke passes):
  ```bash
  source .venv/bin/activate
  python local_pack.py --workers 4    # 6 is fine, but --workers 4 keeps CPU cooler; run detached
  ```
  Run detached so a laptop sleep/lid-close doesn't kill it: `nohup ... > local-pack.log 2>&1 &`. Monitor `tail -f local-pack.log`. With 6-7 cores doing download-overlap + single-threaded torch.load, expect CPU to stay well under saturation; the fiber link is the bottleneck (~2.5-3 h total download for 950 GB).
