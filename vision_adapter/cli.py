"""vision_adapter.cli — staged entrypoint dataset|precompute|pack|train|probe."""
from __future__ import annotations

import argparse
import sys


def dataset_cmd(args: argparse.Namespace) -> int:
    print(f"dataset --out {args.out} (stub — wire to dataset.py)")
    return 0


def precompute_cmd(args: argparse.Namespace) -> int:
    print(f"precompute --data-dir {args.data_dir} (stub)")
    return 0


def pack_cmd(args: argparse.Namespace) -> int:
    print(f"pack --data-dir {args.data_dir} (stub)")
    return 0


def train_cmd(args: argparse.Namespace) -> int:
    cfg = getattr(args, "config", "default")
    print(f"train --config {cfg} (stub)")
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
    p.add_argument("--out", required=True, help="output manifest path")
    p.add_argument("--seed", type=int, default=0, help="python seed")
    p.add_argument("--limit", type=int, default=54000, help="max rows")
    p.add_argument("--upstream-pin", default=None, help="upstream pin")
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
    p.add_argument("--dryrun", action="store_true", help="dry run")
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
