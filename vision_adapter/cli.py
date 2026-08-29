"""vision_adapter.cli — staged entrypoint dataset|precompute|pack|train|probe."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def dataset_cmd(args: argparse.Namespace) -> int:
    from vision_adapter.backends.base import get_backend
    from vision_adapter.data.dataset import build_dataset

    backend = get_backend(args.backend, **({"root": args.out} if args.backend == "local" else {}))
    # For local, out is the output dir; for modal it's also the out dir on Volume.
    # build_dataset takes (backend, out_dir, seed, limit, upstream_pin, dry_run)
    dry_run = getattr(args, "dry_run", False)
    out = Path(args.out)
    path = build_dataset(
        backend,
        out,
        seed=args.seed,
        limit=args.limit,
        upstream_pin=getattr(args, "upstream_pin", None),
        dry_run=dry_run,
    )
    print(f"[dataset] wrote {path} (seed={args.seed}, limit={args.limit})")
    return 0


def precompute_cmd(args: argparse.Namespace) -> int:
    from vision_adapter.backends.base import get_backend
    from vision_adapter.models.precompute import run_precompute

    backend = get_backend(args.backend, **({"root": args.data_dir} if args.backend == "local" else {}))
    run_precompute(
        backend,
        Path(args.data_dir),
        patch_cap=args.patch_cap or 262144,
        device=args.device,
        revision=getattr(args, "revision", None),
    )
    print(f"[precompute] ok --data-dir {args.data_dir} (backend={args.backend})")
    return 0


def pack_cmd(args: argparse.Namespace) -> int:
    from vision_adapter.backends.base import get_backend

    # Pack is backend-agnostic when data_dir is local; Modal uses Volume directly.
    # Delegate to pack_stage if available, otherwise direct pack.
    backend = get_backend(args.backend, **({"root": args.data_dir} if args.backend == "local" else {}))
    try:
        from vision_adapter.data.pack import pack_stage

        pack_stage(backend, Path(args.data_dir), shard_rows=args.shard_rows)
    except ImportError:
        print(f"[pack] pack_stage not yet wired — data-dir {args.data_dir} ready for pack")
    print(f"[pack] ok --data-dir {args.data_dir} --shard-rows {args.shard_rows} (backend={args.backend})")
    return 0


def train_cmd(args: argparse.Namespace) -> int:
    cfg = getattr(args, "config", "default")
    backend_name = getattr(args, "backend", "local")
    dryrun = getattr(args, "dryrun", False)
    data_dir = Path(getattr(args, "data_dir", "data"))
    print(f"[train] --config {cfg} --backend {backend_name} --data-dir {data_dir}" + (" --dryrun" if dryrun else ""))
    if dryrun:
        print("[train] dryrun ok (no GPU, no Modal)")
        return 0
    # Real train path would delegate to vision_adapter.models.train / modal_train
    print("[train] stub — wire to train loop (modal_train._train_impl or local train)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vision-adapter",
        description="Vision-Adapter staged pipeline",
    )
    subs = ap.add_subparsers(dest="cmd", required=True)

    # dataset
    p = subs.add_parser(
        "dataset",
        help="build header-first manifest (ORDER BY image, pinned revisions)",
    )
    p.add_argument("--out", required=True, help="output directory (manifest written to <out>/train_manifest.jsonl)")
    p.add_argument("--seed", type=int, default=0, help="python seed")
    p.add_argument("--limit", type=int, default=54000, help="max rows")
    p.add_argument("--upstream-pin", default=None, help="upstream pin")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="dry-run: generate fake rows, no HF I/O")
    p.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="data backend",
    )
    p.set_defaults(func=dataset_cmd)

    # precompute
    p = subs.add_parser("precompute", help="precompute MoonViT embeddings")
    p.add_argument("--data-dir", default="data", help="data directory")
    p.add_argument("--patch-cap", type=int, default=None, help="max patches per image")
    p.add_argument("--device", default="cuda", help="torch device")
    p.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="data backend",
    )
    p.add_argument("--revision", default=None, help="hf revision pin")
    p.set_defaults(func=precompute_cmd)

    # pack
    p = subs.add_parser("pack", help="pack embeddings into shards")
    p.add_argument("--data-dir", default="data", help="data directory")
    p.add_argument("--shard-rows", type=int, default=1360, help="rows per shard")
    p.add_argument("--hf-repo", default=None, help="hf repo id")
    p.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="data backend",
    )
    p.add_argument("--only", default=None, help="shard range i[:j]")
    p.add_argument("--hf-only", action="store_true", help="push to HF only, skip local shards")
    p.set_defaults(func=pack_cmd)

    # train
    p = subs.add_parser("train", help="train probe")
    p.add_argument("--data-dir", default="data", help="data directory")
    p.add_argument(
        "--config",
        choices=["probe", "colab", "default"],
        default="default",
        help="train config preset",
    )
    p.add_argument("--max-steps", type=int, default=None, help="max training steps")
    p.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="data backend",
    )
    p.add_argument("--dryrun", action="store_true", help="dry run (validate without GPU)")
    p.set_defaults(func=train_cmd)

    # probe (alias for train --config colab)
    p = subs.add_parser("probe", help="alias for train --config colab")
    p.add_argument("--data-dir", default="data", help="data directory")
    p.add_argument("--max-steps", type=int, default=None, help="max training steps")
    p.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="data backend",
    )
    p.add_argument("--dryrun", action="store_true", help="dry run")
    p.set_defaults(func=train_cmd)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # probe alias promotion: probe -> train --config colab
    if getattr(args, "cmd", None) == "probe":
        args.cmd = "train"
        if not getattr(args, "config", None):
            args.config = "colab"
    result = args.func(args)
    if result is None:
        return 0
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
