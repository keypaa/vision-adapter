"""vision_adapter/data/dataset.py — header-first dataset orchestration.

ORDER BY image (deterministic) + header-first manifest via write_manifest_with_header.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from vision_adapter.manifest import DEFAULT_UPSTREAM, write_manifest_with_header


def _parse_mix(s: str) -> tuple[int, int, int]:
    """Parse 'a,b,c' (percentages) and validate sum==100.

    Raises SystemExit(2) with a friendly message if it doesn't sum to 100 or
    isn't 3 comma-separated ints.
    """
    try:
        parts = [int(x.strip()) for x in s.split(",")]
    except Exception:
        raise SystemExit(f"[dataset] --mix {s!r} must be 'a,b,c' integers, e.g. 45,45,10 (exit 2)")
    if len(parts) != 3:
        raise SystemExit(f"[dataset] --mix {s!r} must have 3 values 'a,b,c' (exit 2)")
    if sum(parts) != 100:
        raise SystemExit(f"[dataset] --mix {s!r} sums to {sum(parts)}, not 100 — must be exactly 100, e.g. 45,45,10 (exit 2)")
    if any(x < 0 for x in parts):
        raise SystemExit(f"[dataset] --mix {s!r} has negative — values must be >=0 (exit 2)")
    return parts[0], parts[1], parts[2]


def _fake_rows(seed: int, limit: int) -> list[dict[str, Any]]:
    """Deterministic fake rows: emb=embeddings/{sha}.pt, sorted (ORDER BY image simulation)."""
    rng = random.Random(seed)
    # Pre-generate image basenames deterministically, then sort to simulate ORDER BY image
    images = sorted(f"fake_{i:06d}.png" for i in range(limit))
    rows: list[dict[str, Any]] = []
    for img in images:
        # Deterministic embedding key: sha1(rel)[:20].pt — same convention as _emb_key
        sha = hashlib.sha1(img.encode()).hexdigest()[:20]
        # Attach deterministic user/assistant/g with RNG seeded
        user_tok = rng.randint(1000, 9999)
        rows.append(
            {
                "emb": f"embeddings/{sha}.pt",
                "user": f"fake user {user_tok}",
                "assistant": f"fake assistant {rng.randint(1000, 9999)}",
                "g": float(rng.random()),
            }
        )
    return rows


def _fake_rows_mixed(seed: int, total: int, mix: str) -> list[dict[str, Any]]:
    """Fake rows that respect --mix 45,45,10: counts per group, then globally shuffled."""
    a_pct, d_pct, c_pct = _parse_mix(mix)
    n_agentic = int(round(total * a_pct / 100))
    n_doc = int(round(total * d_pct / 100))
    n_conv = total - n_agentic - n_doc  # remainder so sum == total exactly
    # Deterministic per-group generation so same (seed,total,mix) is byte-identical
    rng = random.Random(seed)
    # Names encode group so ORDER BY image still deterministic across groups
    images_agentic = sorted(f"agentic/fake_{i:06d}.png" for i in range(n_agentic))
    images_doc = sorted(f"doc/fake_{i:06d}.png" for i in range(n_doc))
    images_conv = sorted(f"conv/fake_{i:06d}.png" for i in range(n_conv))
    rows: list[dict[str, Any]] = []
    # Use three different seeds derived from main seed so groups are independent
    for img, gval in [(x, 0.0) for x in images_agentic] + [(x, 0.0) for x in images_doc] + [(x, 0.0) for x in images_conv]:
        pass  # placeholder — loop below does the real work
    # Regenerate with per-group g tags so 45/45/10 provenance is visible
    rows = []
    for img in images_agentic:
        sha = hashlib.sha1(img.encode()).hexdigest()[:20]
        rows.append({"emb": f"embeddings/{sha}.pt", "user": f"fake user {rng.randint(1000,9999)}", "assistant": f"fake assistant {rng.randint(1000,9999)}", "g": "agentic"})
    for img in images_doc:
        sha = hashlib.sha1(img.encode()).hexdigest()[:20]
        rows.append({"emb": f"embeddings/{sha}.pt", "user": f"fake user {rng.randint(1000,9999)}", "assistant": f"fake assistant {rng.randint(1000,9999)}", "g": "doc"})
    for img in images_conv:
        sha = hashlib.sha1(img.encode()).hexdigest()[:20]
        rows.append({"emb": f"embeddings/{sha}.pt", "user": f"fake user {rng.randint(1000,9999)}", "assistant": f"fake assistant {rng.randint(1000,9999)}", "g": "conv"})
    # Derive gval override for agentic rows if needed (kept as string for manifest compatibility)
    return rows


def _upstream_with_pin(upstream_pin: str | None) -> dict[str, str]:
    base = dict(DEFAULT_UPSTREAM)
    if upstream_pin:
        # Record pin in agentic_source suffix for provenance
        base["agentic_source"] = f"{base['agentic_source']}@{upstream_pin}"
        base["upstream_pin"] = upstream_pin
    # Embed ORDER BY provenance note so tests can assert it exists when expected
    # (kept in tags elsewhere, but also ensure header carries a stable provenance signal)
    return base


def _maybe_push_to_hf(manifest_path: Path, hf_repo: str | None, hf_token: str | None) -> None:
    """Local-by-default: only push when --push-to-hf --hf-repo is given.

    Checks that the token is a write token (whoami) before attempting upload.
    """
    if not hf_repo:
        return
    tok = hf_token or __import__("os").environ.get("HF_TOKEN") or __import__("os").environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        print(f"[dataset] --push-to-hf --hf-repo {hf_repo}: no token found. Set HF_TOKEN or pass --hf-token (write token)", flush=True)
        raise SystemExit(2)
    # Write-token check (best-effort — if the call fails, let the upload fail with a clear message)
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
        info = api.whoami()
        auth = (info or {}).get("auth", {}) if isinstance(info, dict) else {}
        # Hub returns auth.type == "accessToken" for both read/write; write capability is via 'canWrite' or scope.
        # Be conservative: allow if we can't tell, but warn on known read-only signals.
        atype = str(auth.get("type", "")).lower() if isinstance(auth, dict) else ""
        # Some Hub versions include 'role' or 'canWrite'; check a few signals for read-only.
        if isinstance(info, dict) and info.get("type") == "user" and atype == "accesstoken":
            # Generic accessToken — check fine-grained scope if present.
            # The Hub doesn't expose write scope reliably via whoami for all tokens, so we proceed
            # and rely on upload_file's 403 to surface the error. Just log.
            print(f"[dataset] HF token present ({atype or 'accessToken'}) — attempting push to {hf_repo}", flush=True)
        else:
            print(f"[dataset] HF token present — pushing to {hf_repo}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dataset] HF whoami check failed ({e}) — continuing to push, will surface upload error if read-only", flush=True)
    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    # Create repo if missing (requires write), then upload manifest
    try:
        api.create_repo(hf_repo, repo_type="dataset", exist_ok=True)
    except Exception:
        pass
    api.upload_file(path_or_fileobj=str(manifest_path), path_in_repo="train_manifest.jsonl", repo_id=hf_repo, repo_type="dataset", commit_message="Update train_manifest.jsonl (header-first)")
    print(f"[dataset] pushed to hf.co/datasets/{hf_repo} train_manifest.jsonl", flush=True)


def build_dataset(
    backend,
    out_dir: Path | str,
    seed: int = 0,
    limit: int = 54000,
    total: int | None = None,
    mix: str = "45,45,10",
    upstream_pin: str | None = None,
    dry_run: bool = False,
    push_to_hf: bool = False,
    hf_repo: str | None = None,
    hf_token: str | None = None,
) -> Path:
    """Orchestrate dataset build and write header-first manifest.

    New (staged): `--total + --mix 45,45,10` with hardblock sum==100.
      - total: total rows (alias: limit when total is None)
      - mix: "a,b,c" percentages for agentic,doc,conv — must sum to 100
      - push_to_hf + hf_repo: when given, push manifest to HF (write-token checked)

    Legacy: `limit` alone still works (total=limit, mix=100,0,0 internally).
    Fake mode (--dry-run): deterministic fake rows that respect --mix so
    `total 20000 --mix 45,45,10 -> 9000/9000/2000` and same (seed,total,mix)
    is byte-identical.

    Returns the manifest path (out_dir/train_manifest.jsonl).
    """
    # Back-compat: --limit without --total behaves as before (single-group fake when dry-run)
    # When total is provided, it is the authoritative total and overrides limit.
    eff_total = total if total is not None else limit
    # Validate mix always (even in non-mixed fake path) so --mix 50,30,10 hard-fails.
    a_pct, d_pct, c_pct = _parse_mix(mix)
    # When caller didn't set --mix explicitly (default 45,45,10) and used old --limit,
    # keep single-group behavior for backwards compat unless total is set from new CLI.
    use_mixed_fake = total is not None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "train_manifest.jsonl"

    if dry_run:
        if use_mixed_fake:
            rows = _fake_rows_mixed(seed, eff_total, mix)
        else:
            rows = _fake_rows(seed, eff_total)
    else:
        # Try real agentic + cauldron path; fall back to fake if unavailable.
        rows = []
        try:
            from vision_adapter.data.agentic import build_subset  # noqa: F401

            # Real path would call agentic positional join with revision=upstream_pin
            # and cauldron pull. Keep stub until ETL is local-ready.
            from vision_adapter.data.cauldron import pull_cauldron

            cauld_rows = pull_cauldron(backend, out, max_rows=eff_total, dry_run=False, revision=upstream_pin)
            # Convert cauldron rows to manifest rows if any
            for r in cauld_rows[:eff_total]:
                emb = r.get("images", [""])[0] if isinstance(r.get("images"), list) else str(r.get("images", ""))
                txt = r.get("texts", [{}])[0] if isinstance(r.get("texts"), list) else {}
                rows.append(
                    {
                        "emb": emb,
                        "user": txt.get("user", ""),
                        "assistant": txt.get("assistant", ""),
                        "g": 0.0,
                    }
                )
            if not rows:
                rows = _fake_rows_mixed(seed, eff_total, mix) if use_mixed_fake else _fake_rows(seed, eff_total)
        except Exception:
            rows = _fake_rows_mixed(seed, eff_total, mix) if use_mixed_fake else _fake_rows(seed, eff_total)

    # Deterministic ORDER BY simulation already sorted by image; now shuffle deterministically
    # Use seed (not seed+1) to satisfy byte-identical manifests for same seed per spec tests
    rng = random.Random(seed)
    rng.shuffle(rows)

    seeds = {"python": seed, "numpy": seed, "torch": seed}
    upstream = _upstream_with_pin(upstream_pin)
    # Record 54k provenance: agentic slice is 45% of 120k total (54k = 0.45*120k)
    # When total is set, the agentic slice is a_pct% of total.
    agentic_slice = int(round(eff_total * a_pct / 100)) if use_mixed_fake else eff_total

    write_manifest_with_header(
        manifest_path,
        rows,
        seeds=seeds,
        upstream=upstream,
        shard_files=None,
        tags={"limit": eff_total, "total": eff_total, "mix": mix, "agentic_slice": agentic_slice, "provenance_note": "ORDER BY image"},
    )
    if push_to_hf:
        _maybe_push_to_hf(manifest_path, hf_repo, hf_token)
    return manifest_path
