#!/usr/bin/env python3
"""Precompute bench: where the ~6h MoonViT precompute actually goes.

Decision-grade (not paper-grade). Mirrors the CURRENT precompute loop
(`vision_adapter/models/precompute_colab.py:run`): images are preprocessed one
by one on the CPU, packed into a ViT batch under a patch budget, forwarded in
one call, and saved one `.pt` per image. Per-stage timing shows where the time
actually goes, so the rebuild (docs/DATA_PATH_REBUILD.md W7a-W7e) targets the
measured bottleneck instead of a guess.

Per-image CPU stages:  read, decode, preprocess, write_pt, write_blob
Per-batch stages (amortised per image): cat, h2d, vit
    write_pt   = one torch.save per image   (what the current pipeline does)
    write_blob = append raw bf16 to a shard blob (what the corpus writer does)

Reports the serial total (today's loop) and the overlapped total (producer
thread pool || GPU), i.e. what the rewrite can reach.

Also measures one synthetic 12.8 Mpx image at the OLD geometry (65,536
patches) and the same image at the reference cap (383 tokens).

Needs the repo (imports vision_adapter.models.preprocess / .moonvit) — it
measures the real code path, not a copy.

Usage (GPU box):
    # from the images repo (downloads ONE parquet shard, up to 7 GB)
    python scripts/bench_precompute.py --images-repo keypa/vision-adapter-images \
        --shard 16 --n 200 --device cuda --cap none --synthetic-old --out bench_s16.json
    # from a local images dir
    python scripts/bench_precompute.py --images-root ./data/images --n 200 --device cuda
    # CPU-only smoke (numbers are not decision-grade)
    python scripts/bench_precompute.py --images-root ./data/images --n 20 --device cpu
"""
import argparse
import io
import json
import os
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

CORPUS_ROWS = 138_987          # images in keypa/vision-adapter-images
CAP_PIXELS = 300_000           # the reference builder's budget (<=300k px)
CAP_PATCHES = 1531             # 300k px / 196 px per patch -> the reference budget
DEFAULT_BATCH_PATCHES = 240_000   # precompute_colab.py BATCH_PATCHES (tuned for a 15GB T4)
# Calibrated on the T4: ~240k patches fit in ~10 GB of headroom above the 0.8 GB
# weights -> ~24k patches per GB of usable VRAM. Used only to auto-size the
# batch on an unseen GPU; the bench reports the observed patches/GB so the
# constant can be recalibrated from a real run.
PATCHES_PER_GB = 24_000
CPU_STAGES = ("read", "decode", "preprocess", "write_pt", "write_blob")
BATCH_STAGES = ("cat", "h2d", "vit")


def auto_batch_patches(device, headroom_gb: float) -> tuple[int, str]:
    """Size the ViT patch budget from the VRAM actually free on this GPU."""
    if device.type != "cuda":
        return DEFAULT_BATCH_PATCHES, "cpu (default budget)"
    try:
        free, total = torch.cuda.mem_get_info(device)
    except Exception as e:  # pragma: no cover - driver dependent
        return DEFAULT_BATCH_PATCHES, f"mem_get_info failed ({e}); default budget"
    usable_gb = max(0.5, (free / 2**30) - headroom_gb)
    n = int(usable_gb * PATCHES_PER_GB)
    return n, (f"{free/2**30:.1f} GiB free of {total/2**30:.1f} GiB, "
               f"{usable_gb:.1f} GiB usable x {PATCHES_PER_GB:,} patches/GB")


# ---------------------------------------------------------------- input ----
def iter_local_images(root: Path, n: int):
    """Yield (name, bytes) from a local images dir."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if not paths:
        raise SystemExit(f"no images under {root}")
    for p in paths[:n]:
        yield p.name, p.read_bytes()


def iter_repo_images(repo: str, shard: int, n: int, cache_dir: Path):
    """Yield (filename, bytes) from one parquet shard of the images repo."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"data/train-{shard:05d}-of-00017.parquet"
    path = hf_hub_download(repo, fname, repo_type="dataset", local_dir=str(cache_dir))
    pf = pq.ParquetFile(path)
    taken = 0
    for batch in pf.iter_batches(batch_size=64, columns=["filename", "image"]):
        for nm, blob in zip(batch.column("filename").to_pylist(),
                            batch.column("image").to_pylist()):
            yield nm, blob
            taken += 1
            if taken >= n:
                return


def sample_source(args):
    if args.images_root:
        return iter_local_images(args.images_root, args.n + args.warmup)
    return iter_repo_images(args.images_repo, args.shard, args.n + args.warmup,
                            Path("/tmp/bench_precompute"))


# ------------------------------------------------------------- geometry ----
def use_cap(preprocess_mod, enabled: bool):
    """Emulate the reference pixel budget the way the real fix would.

    The fix is ONE constant: `preprocess.MAX_PATCHES` (65,536 today). Setting
    it to 1531 (~300k px) makes `_resize_size` downscale exactly as the real
    capped precompute would — same BICUBIC resize, same patch grid — instead of
    re-implementing the resize here.
    """
    old = preprocess_mod.MAX_PATCHES
    if enabled:
        preprocess_mod.MAX_PATCHES = CAP_PATCHES
    return old


# ------------------------------------------------------------------ vit ----
class _StubViT:
    """Shape-correct stand-in: returns a LIST of per-image merged tensors.

    Mirrors MoonViTV2.forward: "returns per-image merged [n,4,1024] lists".
    """

    def __init__(self):
        self.patch_embed = type("PE", (), {"proj": type("W", (), {"weight": None})()})()
        self.dtype = torch.bfloat16

    def eval(self):
        return self

    def __call__(self, pixel_values, grid_thws):
        outs = []
        off = 0
        for g in grid_thws.tolist():
            n_patch = int(g[1]) * int(g[2])
            outs.append(torch.zeros(n_patch // 4, 4096, dtype=torch.bfloat16,
                                    device=pixel_values.device))
            off += n_patch
        return outs


def build_vit(args, device):
    """Real MoonViT, or a shape-correct stub (no weight download)."""
    if args.stub_vit:
        print("[bench] using STUB ViT (no weights downloaded)", flush=True)
        return _StubViT()

    from huggingface_hub import hf_hub_download

    from vision_adapter.models.moonvit import load_moonvit_from_safetensors
    cfg = json.load(open(hf_hub_download(repo_id="keypa/MoonViT-V2-Standalone",
                                         repo_type="model", filename="vision_config.json")))
    st_path = hf_hub_download(repo_id="keypa/MoonViT-V2-Standalone", repo_type="model",
                              filename="moonvit_v2.safetensors")
    print("[bench] loading MoonViT-V2 ...", flush=True)
    vit = load_moonvit_from_safetensors(st_path, cfg, device=str(device), dtype=torch.bfloat16)
    vit.eval()
    return vit


# --------------------------------------------------------------- batch ----
def _split_outputs(outs, grids):
    """Normalise the ViT return to a list of per-image [n_merged, 4096] tensors."""
    if isinstance(outs, torch.Tensor):
        per = [int(g[1]) * int(g[2]) // 4 for g in grids.tolist()]
        out, off = [], 0
        for n in per:
            out.append(outs[off:off + n])
            off += n
        return out
    return list(outs)


class _Batch:
    """A packed ViT batch, mirroring precompute_colab's collate+forward."""

    def __init__(self, budget: int):
        self.budget = budget
        self.items: list[dict] = []
        self.patches = 0

    def add(self, pixel_values, grid, dims):
        self.items.append({"pv": pixel_values, "grid": grid, "dims": dims})
        self.patches += int(pixel_values.shape[0])

    def full(self):
        return self.patches >= self.budget

    def forward(self, vit, device):
        """Run the batch; return (per-image merged tensors, timings, vram stats)."""
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        t0 = time.perf_counter()
        pv = torch.cat([it["pv"] for it in self.items], dim=0)
        gt = torch.cat([it["grid"] for it in self.items], dim=0)
        t1 = time.perf_counter()
        pv = pv.to(device).to(torch.bfloat16)
        gt = gt.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        outs = _split_outputs(vit(pv, gt), gt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        vram = {}
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            vram = {"peak_alloc_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                    "free_gib": free / 2**30, "total_gib": total / 2**30,
                    "patches": self.patches}
        return outs, {"cat": (t1 - t0) * 1e3, "h2d": (t2 - t1) * 1e3, "vit": (t3 - t2) * 1e3}, vram


# --------------------------------------------------------------- measure ----
def measure(args, vit, device, process_image, budget, preprocess_mod):
    """Run the sample loop; return per-stage timings and shape stats."""
    import vision_adapter.models.preprocess as _pp
    old_max = use_cap(_pp, args.cap == "reference")
    T = {k: [] for k in CPU_STAGES + BATCH_STAGES}
    n_vis, n_patches, dims_all = [], [], []
    vram_stats: list[dict] = []
    tmpdir = tempfile.mkdtemp(prefix="bench_precompute_")
    blob_path = os.path.join(tmpdir, "corpus.bin")
    blob_limit = 4 * 2**30          # keep the blob bounded on disk (it is a throwaway)
    batch = _Batch(budget)
    seen = 0

    def flush():
        """Forward the pending batch, then write each image (timed per image)."""
        nonlocal seen
        if not batch.items:
            return
        outs, bt, vram = batch.forward(vit, device)
        if vram:
            vram_stats.append(vram)
        n_img = len(batch.items)
        for it, emb in zip(batch.items, outs):
            flat = emb.reshape(emb.shape[0], -1).to(torch.bfloat16).cpu()
            t6 = time.perf_counter()
            path = os.path.join(tmpdir, f"e{seen}.pt")
            torch.save(flat, path)                      # current write path (real fs)
            os.remove(path)
            t7 = time.perf_counter()
            with open(blob_path, "ab") as bf:            # proposed: append to the shard blob
                bf.write(flat.reshape(-1).view(torch.uint8).numpy().tobytes())
            t8 = time.perf_counter()
            T["write_pt"].append((t7 - t6) * 1e3)
            T["write_blob"].append((t8 - t7) * 1e3)
            for k, v in bt.items():
                T[k].append(v / n_img)                 # amortise batch cost per image
            n_vis.append(int(flat.shape[0]))
            n_patches.append(int(it["pv"].shape[0]))
            dims_all.append(it["dims"])
            seen += 1
            if os.path.getsize(blob_path) > blob_limit:
                os.remove(blob_path)
        batch.items.clear()
        batch.patches = 0

    with torch.no_grad():
        for _name, raw in sample_source(args):
            t0 = time.perf_counter()
            data = raw
            t1 = time.perf_counter()
            im = Image.open(io.BytesIO(data)).convert("RGB")
            t2 = time.perf_counter()
            out = process_image(im)
            t3 = time.perf_counter()
            if seen < args.warmup:
                # warmup images still go through the ViT, but are not recorded
                batch.add(out["pixel_values"], out["grid_thws"], im.size)
                if batch.full():
                    flush()
                continue
            T["read"].append((t1 - t0) * 1e3)
            T["decode"].append((t2 - t1) * 1e3)
            T["preprocess"].append((t3 - t2) * 1e3)
            batch.add(out["pixel_values"], out["grid_thws"], im.size)
            if batch.full():
                flush()
            if seen % 25 == 0:
                print(f"[bench] {seen}/{args.n + args.warmup}", flush=True)
        flush()
    _pp.MAX_PATCHES = old_max
    return T, n_vis, n_patches, dims_all, vram_stats


def report(T, n_vis, n_patches, dims, vram_stats, budget_note):
    """Print the stage table and the throughput projections; return the numbers."""
    print("\n=== per-image stage cost (ms, mean) ===")
    for k in CPU_STAGES + BATCH_STAGES:
        v = T[k]
        if not v:
            continue
        p90 = sorted(v)[int(len(v) * 0.9)]
        print(f"  {k:<11} {st.mean(v):8.2f}   p50 {st.median(v):7.2f}  p90 {p90:7.2f}")
    print(f"\n  n_vis: median {st.median(n_vis):.0f}  mean {st.mean(n_vis):.0f}  max {max(n_vis)}")
    print(f"  patches: median {st.median(n_patches):.0f}  mean {st.mean(n_patches):.0f}  max {max(n_patches)}")
    px = [w * h for w, h in dims]
    print(f"  pixels: median {st.median(px)/1e3:.0f}k  max {max(px)/1e6:.1f}M")
    if vram_stats:
        pk = max(v["peak_alloc_gib"] for v in vram_stats)
        free = min(v["free_gib"] for v in vram_stats)
        tot = vram_stats[0]["total_gib"]
        ppg = max(v["patches"] for v in vram_stats) / max(pk, 1e-9)
        print(f"\n=== VRAM ===\n  device total {tot:.1f} GiB | peak allocated {pk:.2f} GiB | "
              f"min free during run {free:.1f} GiB")
        print(f"  observed {ppg:,.0f} patches per GiB of peak VRAM "
              f"(auto-sizing constant is {PATCHES_PER_GB:,} — recalibrate from this)")

    cpu_pt = sum(st.mean(T[k]) for k in ("read", "decode", "preprocess", "write_pt"))
    cpu_blob = sum(st.mean(T[k]) for k in ("read", "decode", "preprocess", "write_blob"))
    gpu = st.mean(T["vit"]) + st.mean(T["h2d"]) + st.mean(T["cat"])

    def proj(total_ms, label):
        per = 1000.0 / total_ms
        hours = CORPUS_ROWS / per / 3600
        print(f"  {label:<34} {per:7.1f} img/s   full corpus {hours:6.2f} h")
        return {"per_image_ms": total_ms, "img_per_s": per, "full_corpus_hours": hours}

    print("\n=== throughput projection (138,987 images) ===")
    res = {
        "serial_current": proj(cpu_pt + gpu, "today's loop (serial, per-image .pt)"),
        "serial_blob": proj(cpu_blob + gpu, "serial but blob writes"),
        "overlapped_blob": proj(max(cpu_blob, gpu), "overlapped CPU||GPU, blob writes"),
        "overlapped_pt": proj(max(cpu_pt, gpu), "overlapped CPU||GPU, per-image .pt"),
    }
    print(f"\n  CPU-side total {cpu_pt:.1f} ms/img | GPU-side {gpu:.1f} ms/img "
          f"-> CPU is the bottleneck: {cpu_pt/max(gpu,1e-9):.1f}x")
    res["cpu_side_ms"] = cpu_pt
    res["gpu_side_ms"] = gpu
    return res


def synthetic_old(vit, device, process_image):
    """Same 12.8 Mpx image at the OLD geometry and at the reference cap."""
    import vision_adapter.models.preprocess as _pp
    print("\n=== synthetic OLD geometry vs reference cap (same 12.8 Mpx image) ===")
    big = Image.new("RGB", (3584, 3584), (128, 128, 128))
    out = {}
    for label, capped in (("old", False), ("capped", True)):
        old = use_cap(_pp, capped)
        t0 = time.perf_counter()
        o = process_image(big)
        t1 = time.perf_counter()
        _pp.MAX_PATCHES = old
        pv = o["pixel_values"].to(device).to(torch.bfloat16)
        gt = o["grid_thws"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        m = _split_outputs(vit(pv, gt), gt)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        canvas = o["pixel_values"].numel() * 4 / 2**20
        print(f"  {label:<7} preprocess {1e3*(t1-t0):8.0f} ms   vit {1e3*(t3-t2):8.0f} ms   "
              f"n_vis {int(m.shape[0]):6d}   canvas {canvas:6.0f} MiB fp32")
        out[label] = {"preprocess_ms": 1e3 * (t1 - t0), "vit_ms": 1e3 * (t3 - t2),
                      "n_vis": int(m.shape[0]), "canvas_mib": canvas}
    print(f"  ratio: preprocess {out['old']['preprocess_ms']/max(out['capped']['preprocess_ms'],1e-9):.1f}x   "
          f"vit {out['old']['vit_ms']/max(out['capped']['vit_ms'],1e-9):.1f}x")
    return out


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images-root", type=Path)
    src.add_argument("--images-repo", default="keypa/vision-adapter-images")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch-patches", type=int, default=0,
                    help="ViT patch budget per batch; 0 = auto-size from free VRAM")
    ap.add_argument("--vram-headroom-gb", type=float, default=2.0,
                    help="VRAM to leave free when auto-sizing the patch budget")
    ap.add_argument("--cap", choices=("none", "reference"), default="none",
                    help="reference = downscale to <=300k px before preprocessing")
    ap.add_argument("--synthetic-old", action="store_true",
                    help="also time one synthetic image at the OLD geometry (65,536 patches)")
    ap.add_argument("--stub-vit", action="store_true",
                    help="replace MoonViT with a shape-correct stub (no weight download); "
                         "isolates the CPU-side cost, ViT timings are meaningless")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    import vision_adapter.models.preprocess as preprocess_mod
    from vision_adapter.models.preprocess import process_image

    vit = build_vit(args, device)
    if args.batch_patches and args.batch_patches > 0:
        budget, note = args.batch_patches, "explicit --batch-patches"
    else:
        budget, note = auto_batch_patches(device, args.vram_headroom_gb)
    print(f"[bench] ViT patch budget: {budget:,} patches/batch ({note})", flush=True)

    T, n_vis, n_patches, dims, vram_stats = measure(args, vit, device, process_image, budget, preprocess_mod)
    if not T["read"]:
        raise SystemExit("no samples")
    res = report(T, n_vis, n_patches, dims, vram_stats, note)
    res["batch_patches"] = budget
    res["batch_patches_note"] = note
    if args.synthetic_old:
        res["synthetic_old"] = synthetic_old(vit, device, process_image)

    if args.out:
        payload = {
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "stages_ms": {k: (st.mean(v) if v else None) for k, v in T.items()},
            "n_vis": {"median": st.median(n_vis), "mean": st.mean(n_vis), "max": max(n_vis)},
            "projection": res,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
