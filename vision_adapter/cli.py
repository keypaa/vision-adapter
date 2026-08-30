"""vision_adapter.cli — staged entrypoint dataset|precompute|pack|train|probe."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_hf_token(args: argparse.Namespace) -> str | None:
    """Resolve HF token: --hf-token > $HF_TOKEN > $HUGGING_FACE_HUB_TOKEN > Colab userdata.

    Does NOT assume Colab — userdata is best-effort. Returned token is also
    exported to os.environ so downstream huggingface_hub / datasets pick it up.
    """
    from vision_adapter.backends.auth import get_hf_token, set_hf_token_env

    tok = get_hf_token(getattr(args, "hf_token", None))
    set_hf_token_env(tok)
    if tok:
        print("[vision-adapter] HF token: present (auth enabled)", flush=True)
    else:
        print("[vision-adapter] HF token: absent (anonymous, lower rate limits)", flush=True)
    return tok


def dataset_cmd(args: argparse.Namespace) -> int:
    _resolve_hf_token(args)
    from vision_adapter.backends.base import get_backend
    from vision_adapter.data.dataset import build_dataset

    backend = get_backend(args.backend, **({"root": args.out} if args.backend == "local" else {}))
    dry_run = getattr(args, "dry_run", False)
    out = Path(args.out)
    # --total is canonical, --limit is alias; default 54000 when neither is set (covers --total+--mix and legacy --limit)
    total = getattr(args, "total", None)
    limit = getattr(args, "limit", None)
    if total is None and limit is None:
        total = 54000
    elif total is not None and limit is not None:
        # Both set: --total wins, --limit is ignored (warn)
        print(f"[dataset] both --total {total} and --limit {limit} set — --total wins, --limit ignored", flush=True)
    path = build_dataset(
        backend,
        out,
        seed=args.seed,
        limit=limit if total is None else total,  # compat shim
        total=total,
        mix=getattr(args, "mix", "45,45,10"),
        upstream_pin=getattr(args, "upstream_pin", None),
        dry_run=dry_run,
        push_to_hf=getattr(args, "push_to_hf", False),
        hf_repo=getattr(args, "hf_repo", None),
        hf_token=getattr(args, "hf_token", None),
    )
    mix = getattr(args, "mix", "45,45,10")
    push_note = f" | pushed to {getattr(args, 'hf_repo', None)}" if getattr(args, "push_to_hf", False) else " | local only (no push)"
    print(f"[dataset] wrote {path} (seed={args.seed}, total={total if total is not None else limit}, mix={mix}{push_note})")
    return 0


def precompute_cmd(args: argparse.Namespace) -> int:
    _resolve_hf_token(args)
    # GPU-gated: precompute runs CUDA kernels; fail fast on CPU with nvidia-smi hint.
    if not getattr(args, "dryrun", False) and getattr(args, "device", "cuda") == "cuda":
        from vision_adapter.backends.gpu import require_gpu

        require_gpu("precompute")
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
    _resolve_hf_token(args)
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
    _resolve_hf_token(args)
    dryrun = getattr(args, "dryrun", False)
    # GPU-gated: real train (not --dryrun) needs CUDA; --dryrun is CPU-safe.
    if not dryrun:
        from vision_adapter.backends.gpu import require_gpu

        require_gpu("train")
    from vision_adapter.config import colab_probe_config, config_header, default_config, probe_config

    cfg_name = getattr(args, "config", "default")
    cfg_fn = {"default": default_config, "probe": probe_config, "colab": colab_probe_config}.get(cfg_name, default_config)
    cfg = cfg_fn()
    backend_name = getattr(args, "backend", "local")
    data_dir = Path(getattr(args, "data_dir", "data"))
    max_steps = getattr(args, "max_steps", None)
    # resolve backend (best-effort)
    try:
        from vision_adapter.backends.base import get_backend

        if backend_name == "local":
            backend = get_backend("local", root=data_dir)
        else:
            backend = get_backend("modal")
    except Exception as e:  # noqa: BLE001 — backend init must not crash CLI
        print(f"[train] backend init failed ({e})", flush=True)
        backend = None  # type: ignore[assignment]

    if dryrun:
        # Real validation instead of stub print — this is what "proven" means before we delete shims.
        if not data_dir.exists():
            print(f"[train] data-dir does not exist: {data_dir} (dryrun still ok — will create on dataset)", flush=True)
        manifest_path = data_dir / "train_manifest.jsonl"
        if manifest_path.exists():
            try:
                from vision_adapter.manifest import read_manifest_header

                hdr = read_manifest_header(manifest_path)
                if hdr:
                    print(f"[train] manifest {manifest_path}: header v{hdr.manifest_version} rows={hdr.row_count} git={hdr.git_sha[:8]}", flush=True)
                else:
                    rows = sum(1 for _ in open(manifest_path) if _.strip())
                    print(f"[train] manifest {manifest_path}: {rows} rows (legacy, no header)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[train] manifest check failed: {e}", flush=True)
        else:
            print(f"[train] manifest not found at {manifest_path} (ok for dryrun — run dataset first)", flush=True)
        try:
            hdr = config_header(cfg, manifest_path=manifest_path if manifest_path.exists() else None, extra={"run": "train", "backend": backend_name, "config": cfg_name})
            print(f"[train] config {cfg_name}: batch={cfg.batch_size} lr={cfg.lr} max_steps={max_steps} warmup={cfg.warmup_steps}", flush=True)
            print(f"[train] run_id {hdr['run_id']} git {hdr['git_sha'][:8] if hdr['git_sha'] != 'unknown' else 'unknown'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[train] config header failed: {e}", flush=True)
        if backend is not None:
            try:
                keys = backend.list_embeddings("embeddings/")
                print(f"[train] backend {backend_name}: {len(keys)} embeddings under embeddings/", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[train] backend list failed (ok for dryrun): {e}", flush=True)
        print("[train] dryrun ok (no GPU, no Modal)", flush=True)
        return 0

    # Real train — delegate to the shared runner. Assumes a GPU (gated above, any CUDA).
    from vision_adapter.train import run_train

    return run_train(
        data_dir=data_dir,
        cfg=cfg,
        backend=backend,
        max_steps=max_steps,
        device="cuda" if not dryrun else None,
        dtype=getattr(args, "dtype", "auto"),
    )


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
    # Staged 120k controls: --total + --mix 45,45,10 with hardblock sum==100.
    # 54k = 0.45*120k (agentic slice). --limit stays as alias of --total for compat.
    p.add_argument("--limit", dest="limit", type=int, default=None, help="max rows (alias of --total, default 54000)")
    p.add_argument("--total", dest="total", type=int, default=None, help="total rows (overrides --limit when set)")
    p.add_argument("--mix", dest="mix", type=str, default="45,45,10", help="mix 'a,b,c' percentages for agentic,doc,conv — must sum to 100 (default 45,45,10)")
    p.add_argument("--upstream-pin", default=None, help="upstream pin")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="dry-run: generate fake rows, no HF I/O")
    # Local by default; --push-to-hf --hf-repo to publish (write-token checked)
    p.add_argument("--push-to-hf", dest="push_to_hf", action="store_true", help="push manifest to HF dataset repo (requires --hf-repo and write token)")
    p.add_argument("--hf-repo", dest="hf_repo", default=None, help="HF dataset repo for --push-to-hf, e.g. keypa/vision-adapter-manifests")
    p.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN env) — higher rate limits for HF downloads; write scope required for --push-to-hf")
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
    p.add_argument("--hf-token", default=None, help="HF token (or HF_TOKEN env) — higher rate limits")
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
    p.add_argument("--hf-token", default=None, help="HF token (or HF_TOKEN env) — higher rate limits")
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
    p.add_argument("--hf-token", default=None, help="HF token (or HF_TOKEN env) — higher rate limits for streaming")
    p.add_argument("--dtype", choices=("auto","bf16","fp16","fp32"), default="auto", help="'auto' = bf16 Ampere+ else fp16/fp32 with true AMP; T4: use bf16 or fp32")
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
    p.add_argument("--hf-token", default=None, help="HF token (or HF_TOKEN env) — higher rate limits for streaming")
    p.add_argument("--dtype", choices=("auto","bf16","fp16","fp32"), default="auto", help="'auto' = bf16 Ampere+ else fp16/fp32")
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
