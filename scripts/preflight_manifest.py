#!/usr/bin/env python3
"""Pre-flight: can the trainer's geometry contract hold on the live manifest?

The collate validates every row that carries a measured grid (vision_adapter/
core.py::make_collate): the grid's placeholder count must equal the row's
visual token count, and the grid must be one the position code can express.
A row that fails either raises — which, in a 4,000-step probe, means the run
dies after the data transfer has already been paid for.

This runs that same check over the live manifest against the real key index,
so a mismatch is found in seconds instead of mid-probe. It downloads only a
slice of the manifest (Range) and no corpus data.

Usage:
    python scripts/preflight_manifest.py
    python scripts/preflight_manifest.py --mib 40
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vision_adapter.core import grid_token_count, placeholder_count

MANIFEST_REPO = "keypa/vision-adapter-manifests"
MANIFEST_FILE = "train_manifest_grids.jsonl"
MIB = 2**20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=20, help="how much of the manifest to read")
    ap.add_argument("--key-index", type=Path,
                    default=Path("logs/key_index_cache.json"),
                    help="the key index (key -> n_vis) to check against")
    args = ap.parse_args()

    if not args.key_index.is_file():
        print(f"[preflight] no key index at {args.key_index} — run the trainer once "
              f"(or fetch the index) to build it")
        return 2
    idx = json.load(open(args.key_index))["keys"]

    req = urllib.request.Request(
        f"https://huggingface.co/datasets/{MANIFEST_REPO}/resolve/main/{MANIFEST_FILE}",
        headers={"Range": f"bytes=0-{args.mib * MIB}"})
    raw = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
    print(f"[preflight] read {len(raw)/MIB:.0f} MiB of {MANIFEST_FILE}")

    n = no_grid = 0
    count_mismatch: list = []
    enum_mismatch: list = []
    odd_dims: list = []
    multi_frame: list = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except Exception:
            break
        if o.get("type") == "manifest_header":
            continue
        g = o.get("grid_thw")
        if not g:
            no_grid += 1
            continue
        v = idx.get(o.get("emb"))
        if not v or not v.get("n_vis"):
            continue
        n += 1
        nv = v["n_vis"]
        g = [int(x) for x in g]
        c = placeholder_count(torch.tensor(g))
        e = grid_token_count(g)
        if c != nv:
            count_mismatch.append((g, c, nv))
        if e != nv:
            enum_mismatch.append((g, e, nv))
        if g[1] % 2 or g[2] % 2:
            odd_dims.append(g)
        if g[0] != 1:
            multi_frame.append(g)

    print(f"[preflight] rows checked against the key index: {n}  (no grid_thw: {no_grid})")
    print(f"[preflight]   placeholder_count != n_vis : {len(count_mismatch)}  {count_mismatch[:3]}")
    print(f"[preflight]   grid_token_count != n_vis : {len(enum_mismatch)}  {enum_mismatch[:3]}")
    print(f"[preflight]   odd dims (h or w not even): {len(odd_dims)}  {odd_dims[:3]}")
    print(f"[preflight]   multi-frame (t != 1)       : {len(multi_frame)}  {multi_frame[:3]}")
    bad = count_mismatch or enum_mismatch or odd_dims or multi_frame
    if n == 0:
        print("[preflight] INCONCLUSIVE: no rows could be checked (is the key index for "
              "this corpus?)")
        return 2
    if bad:
        print("[preflight] FAIL: the collate would raise on these rows — the probe would "
              "die after paying for the data transfer")
        return 1
    print("[preflight] PASS: the geometry contract holds on every checked row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
