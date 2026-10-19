#!/usr/bin/env python3
"""Precompute bench: where the ~6h MoonViT precompute actually goes.

Decision-grade (not paper-grade). Measures the per-stage cost of the CURRENT
precompute path (`vision_adapter/models/precompute_colab.py:run`) so the
rebuild (docs/DATA_PATH_REBUILD.md W7a-W7e) is aimed at the real bottleneck
instead of a guess.

Stages timed per image:
    read      bytes -> memory (file or parquet row)
    decode    PIL open + convert("RGB")
    preprocess resize + pad + normalize + patchify   (CPU, vision_adapter/models/preprocess.py)
    h2d       pixel_values -> GPU
    vit       MoonViT-V2 forward -> merged tokens
    write_pt  torch.save per image          (what the current pipeline does)
    write_blob append raw bf16 to a buffer  (what the new corpus writer does)

Reports the serial total (what today's loop pays) AND the overlapped total
(producer thread pool || GPU), i.e. what the rewrite can reach.

Also measures a synthetic large image at the OLD geometry (65,536 patches) and
the same image at the reference cap (383 tokens) to quantify why the 6h
happened and what the cap buys.

Needs the repo (imports vision_adapter.models.preprocess / .moonvit) — it
measures the real code path, not a copy.

Usage (GPU box):
    # from a local images dir
    python scripts/bench_precompute.py --images-root ./data/images --n 200 --device cuda
    # from the images repo (downloads ONE parquet shard, ~320 MiB)
    python scripts/bench_precompute.py --images-repo keypa/vision-adapter-images --n 200 --device cuda
    # CPU-only smoke (numbers are not decision-grade)
    python scripts/bench_precompute.py --images-root ./data/images --n 20 --device cpu
"""
import argparse
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

CORPUS_ROWS = 138_987          # images in keypa/vision-adapter-images
CAP_PIXELS = 300_000           # the reference builder's budget (<=300k px)
STAGES = ("read", "decode", "preprocess", "h2d", "vit", "write_pt", "write_blob")


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
def cap_to_reference(im: Image.Image) -> Image.Image:
    """Downscale to <=300k px, floor dims to 28-multiples (agentic.py:275 contract)."""
    import math
    w, h = im.size
    scale = min(1.0, math.sqrt(CAP_PIXELS / max(1, w * h)))
    nw = max(28, int(w * scale) // 28 * 28)
    nh = max(28, int(h * scale) // 28 * 28)
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.LANCZOS)
    return im


# ------------------------------------------------------------------ vit ----
def build_vit(args, device):
    """Real MoonViT, or a shape-correct stub (no weight download)."""
    if args.stub_vit:
        class _StubViT:
            """[n_patches, 3, 14, 14] -> [n_merged, 4096]."""
            def __init__(self):
                self.patch_embed = type("PE", (), {"proj": type("W", (), {"weight": None})()})()
                self.dtype = torch.bfloat16

            def eval(self):
                return self

            def __call__(self, pixel_values, grid_thws):
                n = int(pixel_values.shape[0])
                return torch.zeros(n // 4, 4096, dtype=torch.bfloat16,
                                   device=pixel_values.device)
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


# --------------------------------------------------------------- measure ----
def measure(args, vit, device, process_image):
    """Run the sample loop; return per-stage timings and shape stats."""
    T = {k: [] for k in STAGES}
    n_vis, n_patches, dims = [], [], []
    blob_buf = io.BytesIO()
    i = 0
    with torch.no_grad():
        for _name, raw in sample_source(args):
            t0 = time.perf_counter()
            data = raw
            t1 = time.perf_counter()
            im = Image.open(io.BytesIO(data)).convert("RGB")
            t2 = time.perf_counter()
            if args.cap == "reference":
                im = cap_to_reference(im)
            out = process_image(im)
            t3 = time.perf_counter()
            pv = out["pixel_values"].to(device).to(torch.bfloat16)
            gt = out["grid_thws"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.perf_counter()
            merged = vit(pv, gt)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t5 = time.perf_counter()
            flat = merged.reshape(merged.shape[0], -1).to(torch.bfloat16).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t6 = time.perf_counter()
            buf = io.BytesIO()
            torch.save(flat, buf)                      # current write path
            t7 = time.perf_counter()
            blob_buf.write(flat.reshape(-1).view(torch.uint8).numpy().tobytes())
            t8 = time.perf_counter()

            i += 1
            if i <= args.warmup:
                continue
            T["read"].append((t1 - t0) * 1e3)
            T["decode"].append((t2 - t1) * 1e3)
            T["preprocess"].append((t3 - t2) * 1e3)
            T["h2d"].append((t4 - t3) * 1e3)
            T["vit"].append((t5 - t4) * 1e3)
            T["write_pt"].append((t7 - t6) * 1e3)
            T["write_blob"].append((t8 - t7) * 1e3)
            n_vis.append(int(merged.shape[0]))
            n_patches.append(int(out["pixel_values"].shape[0]))
            dims.append(im.size)
            if (i - args.warmup) % 25 == 0:
                print(f"[bench] {i}/{args.n + args.warmup}", flush=True)
    return T, n_vis, n_patches, dims


def report(T, n_vis, n_patches, dims):
    """Print the stage table and the throughput projections; return the numbers."""
    print("\n=== per-image stage cost (ms, mean) ===")
    for k, v in T.items():
        p90 = sorted(v)[int(len(v) * 0.9)]
        print(f"  {k:<11} {st.mean(v):8.2f}   p50 {st.median(v):7.2f}  p90 {p90:7.2f}")
    print(f"\n  n_vis: median {st.median(n_vis):.0f}  mean {st.mean(n_vis):.0f}  max {max(n_vis)}")
    print(f"  patches: median {st.median(n_patches):.0f}  mean {st.mean(n_patches):.0f}  max {max(n_patches)}")
    px = [w * h for w, h in dims]
    print(f"  pixels: median {st.median(px)/1e3:.0f}k  max {max(px)/1e6:.1f}M")

    cpu_pt = sum(st.mean(T[k]) for k in ("read", "decode", "preprocess", "write_pt"))
    cpu_blob = sum(st.mean(T[k]) for k in ("read", "decode", "preprocess", "write_blob"))
    gpu = st.mean(T["vit"]) + st.mean(T["h2d"])

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
          f"-> CPU is the bottleneck: {cpu_pt/gpu:.1f}x")
    res["cpu_side_ms"] = cpu_pt
    res["gpu_side_ms"] = gpu
    return res


def synthetic_old(vit, device, process_image):
    """Same 12.8 Mpx image at the OLD geometry and at the reference cap."""
    print("\n=== synthetic OLD geometry vs reference cap (same 12.8 Mpx image) ===")
    big = Image.new("RGB", (3584, 3584), (128, 128, 128))
    out = {}
    for label, im in (("old", big), ("capped", cap_to_reference(big))):
        t0 = time.perf_counter()
        o = process_image(im)
        t1 = time.perf_counter()
        pv = o["pixel_values"].to(device).to(torch.bfloat16)
        gt = o["grid_thws"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        m = vit(pv, gt)
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
    from vision_adapter.models.preprocess import process_image

    vit = build_vit(args, device)
    T, n_vis, n_patches, dims = measure(args, vit, device, process_image)
    if not T["read"]:
        raise SystemExit("no samples")
    res = report(T, n_vis, n_patches, dims)
    if args.synthetic_old:
        res["synthetic_old"] = synthetic_old(vit, device, process_image)

    if args.out:
        payload = {
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "stages_ms": {k: st.mean(v) for k, v in T.items()},
            "n_vis": {"median": st.median(n_vis), "mean": st.mean(n_vis), "max": max(n_vis)},
            "projection": res,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
