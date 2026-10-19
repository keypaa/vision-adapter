#!/usr/bin/env python3
"""Filter a train manifest: drop the rows that cannot be trained on.

Two classes of row are dropped (see docs/DATA_PATH_REBUILD.md W0c):

* **long-answer rows** — the source recipe states that long descriptive
  answers *prevent* the grokking event. Our mix carries 445 HTML-generation
  targets ("Synthesize the HTML to emulate this website's layout") and ~200
  rationale-style answers ("Answer: car.\\nRationale: ...").
* optionally, rows whose **answer is HTML** at all, detected by the target
  starting with "<" (catches the HTML-generation slice without hard-coding a
  subset name).

The output is a header-first manifest (same schema as the input), so the
trainer consumes it unchanged. The dropped keys are printed and can be fed to
a corpus re-pack later.

Usage:
    python scripts/filter_manifest.py --manifest data/train_manifest_grids.jsonl \
        --out data/train_manifest_filtered.jsonl --max-answer-words 20
    # the manifest is not on the box yet -> fetch it from the manifests repo first
    python scripts/filter_manifest.py --manifest data/train_manifest_grids.jsonl \
        --out data/train_manifest_filtered.jsonl --max-answer-words 20 --fetch
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def looks_like_html(answer: str) -> bool:
    a = answer.strip().lower()
    return a.startswith("<") or a.startswith("<!doctype")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-answer-words", type=int, default=20,
                    help="drop rows whose answer is longer than this (0 = keep all)")
    ap.add_argument("--drop-html", action="store_true", default=True,
                    help="drop rows whose answer is HTML (the grokking blocker)")
    ap.add_argument("--keep-html", dest="drop_html", action="store_false")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch the manifest from the manifests repo if it is missing")
    args = ap.parse_args()

    if args.fetch and not args.manifest.exists():
        from huggingface_hub import hf_hub_download

        from vision_adapter.data.stream import MANIFEST_FILE, MANIFEST_REPO

        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(MANIFEST_REPO, MANIFEST_FILE, repo_type="dataset",
                            local_dir=str(args.manifest.parent))
        print(f"[filter] fetched manifest -> {p}")

    raw = [json.loads(line) for line in open(args.manifest) if line.strip()]
    header = raw[0] if raw and raw[0].get("type") == "manifest_header" else None
    rows = raw[1:] if header else raw

    kept, dropped = [], []
    reasons = collections.Counter()
    for r in rows:
        ans = str(r.get("assistant", ""))
        w = len(ans.split())
        if args.drop_html and looks_like_html(ans):
            dropped.append(r)
            reasons["html_target"] += 1
            continue
        if args.max_answer_words and w > args.max_answer_words:
            dropped.append(r)
            reasons["long_answer"] += 1
            continue
        kept.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        if header:
            hdr = dict(header)
            hdr["row_count"] = len(kept)
            tags = dict(hdr.get("tags") or {})
            tags["filtered_from"] = args.manifest.name
            tags["dropped_rows"] = len(dropped)
            tags["drop_reasons"] = dict(reasons)
            hdr["tags"] = tags
            f.write(json.dumps(hdr) + "\n")
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"[filter] {len(rows)} -> {len(kept)} rows (dropped {len(dropped)}: {dict(reasons)})")
    by_g = collections.Counter(r.get("g") for r in dropped)
    print(f"[filter] dropped by group: {dict(by_g)}")
    kept_g = collections.Counter(r.get("g") for r in kept)
    print(f"[filter] kept by group:    {dict(kept_g)}")
    print(f"[filter] wrote {args.out}")
    if dropped:
        keys = [r.get("emb") for r in dropped]
        Path(str(args.out) + ".dropped_keys.json").write_text(json.dumps(keys))
        print(f"[filter] dropped keys -> {args.out}.dropped_keys.json ({len(keys)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
