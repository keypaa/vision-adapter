#!/usr/bin/env python3
"""Pack the val split, one parquet per group, so a probe costs one fetch not 21.

The disjoint manifest (`train_manifest_val_disjoint.jsonl`) already guarantees
that no val row's `emb` key appears in the train manifest — verified by an
assert when it was built (scripts/rebuild_val_disjoint.py). So there is no
disjointness work to do here. What is broken is the *plan*: run_train rebuilds
the val plan with `excluded_shards=set(plan.keys())`, which drops 1272 rows to
77 spread over 21 shards. Streaming 21 shards for 77 rows costs ~10 min per
probe against ~2 min of training between probes.

**One pack per group**, not one blended pack. The corpus is 73% agentic
(measured on the train manifest, not the 45/45/10 of the dataset card), so a
blended val_loss can hold steady while one group stops learning — and the
Baseten threshold only means anything on agentic. Splitting also exposes the
shortfall instead of absorbing it: the val split holds ~60 doc rows against a
quota of 270.

Within each pack the corpus bucket mix is preserved. RAM-bounded: one row group
at a time, two columns at a time.

Usage:
    python scripts/pack_val.py --n 1000
    python scripts/pack_val.py --n 1000 --dry-run   # quotas only
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The six n_vis buckets from the dataset card.
BUCKETS = [(0, 100), (101, 500), (501, 1000), (1001, 2000), (2001, 4900), (4901, 10**9)]


def bucket_of(n_vis: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= n_vis <= hi:
            return i
    return len(BUCKETS) - 1


def bucket_name(n_vis: int) -> str:
    lo, hi = BUCKETS[bucket_of(n_vis)]
    return f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"


def key_index(args) -> dict:
    from vision_adapter.data.stream import load_key_index

    for cand in (args.key_index, "emb_cache/cache", "."):
        if not cand:
            continue
        p = Path(cand)
        if p.is_dir():
            hits = sorted(p.glob("key_index_cache*.json"))
            if hits:
                idx, ok = load_key_index(str(hits[0]))
                if ok:
                    print(f"[pack-val] key index: {hits[0].name} ({len(idx)} entries)")
                    return idx
        elif p.is_file():
            idx, ok = load_key_index(str(p))
            if ok:
                return idx
    raise SystemExit("[pack-val] no key index found — pass --key-index")


def quotas_from_corpus(train_rows, index, n_total: int) -> tuple[dict, dict]:
    """Proportional to the TRAIN corpus, not to the val split.

    The val split is small (1272 rows) so its own marginals are noisy; the
    corpus is the thing the model is being trained to match.
    """
    g_count: Counter = Counter()
    b_count: Counter = Counter()
    seen = set()
    for r in train_rows:
        e = r.get("emb")
        if not e or e in seen:
            continue
        seen.add(e)
        loc = index.get(e)
        if not loc or len(loc) != 3:
            continue
        g_count[r.get("g", "?")] += 1
        b_count[bucket_of(int(loc[2]))] += 1
    tot_g, tot_b = sum(g_count.values()), sum(b_count.values())
    if not tot_g:
        raise SystemExit("[pack-val] no indexable train rows")
    g_q = {g: max(1, round(n_total * c / tot_g)) for g, c in g_count.items() if c}
    b_q = {b: max(1, round(n_total * c / tot_b)) for b, c in b_count.items() if c}
    print(f"[pack-val] corpus g mix  : {dict(g_count)}")
    print(f"[pack-val] corpus buckets: "
          f"{ {bucket_name(BUCKETS[b][0]): c for b, c in sorted(b_count.items())} }")
    return g_q, b_q


def pick_per_group(val_rows, index, g_quota, bucket_share, seed=0):
    """One pack per group, each shaped to the corpus bucket mix.

    Splitting the val is the point: the corpus is 73% agentic, so a blended
    val_loss can hold steady while one group stops learning — and the Baseten
    threshold only means anything on agentic. Separate packs also expose the
    measured shortfall instead of absorbing it: the val split holds ~60 doc
    rows against a quota of 270.

    Within each group the corpus bucket mix is preserved, so a doc pack still
    spans small and large images.
    """
    import random

    rng = random.Random(seed)
    by_group: dict[str, list] = defaultdict(list)
    seen: set[str] = set()
    for r in val_rows:
        e = r.get("emb")
        if not e or e in seen:
            continue
        loc = index.get(e)
        if not loc or len(loc) != 3:
            continue
        seen.add(e)
        by_group[r.get("g", "?")].append(r)

    packs: dict[str, list] = {}
    for g, want_total in g_quota.items():
        pool = by_group.get(g, [])
        if not pool or want_total <= 0:
            packs[g] = []
            continue
        cells: dict[int, list] = defaultdict(list)
        for r in pool:
            cells[bucket_of(int(index[r["emb"]][2]))].append(r)

        chosen = []
        for b, rows in sorted(cells.items()):
            take = min(len(rows), round(want_total * bucket_share.get(b, 0.0)))
            rng.shuffle(rows)
            chosen.extend(rows[:take])
        # bucket order, so the streamed batches stay size-homogeneous
        chosen.sort(key=lambda r: index[r["emb"]][2])
        packs[g] = chosen
    return packs


def pack(args) -> int:
    import pyarrow.parquet as pq

    from huggingface_hub import hf_hub_download

    from vision_adapter.data.pack import pack_rows
    from vision_adapter.data.stream import _download_shard_hf_transfer
    from vision_adapter.train import VAL_MANIFEST_FILE, _val_rows_from_file

    mp = args.manifest
    if not mp or not Path(mp).is_file():
        mp = hf_hub_download("keypa/vision-adapter-manifests", VAL_MANIFEST_FILE,
                             repo_type="dataset")
    val_rows = _val_rows_from_file(mp)
    print(f"[pack-val] val manifest: {len(val_rows)} rows")

    tp = args.train_manifest
    if not tp or not Path(tp).is_file():
        tp = hf_hub_download("keypa/vision-adapter-manifests", "train_manifest_grids.jsonl",
                             repo_type="dataset")
    train_rows = _val_rows_from_file(tp)

    index = key_index(args)
    g_q, b_q = quotas_from_corpus(train_rows, index, args.n)
    b_tot = sum(b_q.values()) or 1
    bucket_share = {b: c / b_tot for b, c in b_q.items()}
    packs = pick_per_group(val_rows, index, g_q, bucket_share, seed=args.seed)

    # the invariant the whole script exists to preserve
    train_embs = {r["emb"] for r in train_rows if r.get("emb")}
    for g, rows in packs.items():
        overlap = [r["emb"] for r in rows if r["emb"] in train_embs]
        assert not overlap, (f"val pack '{g}' overlaps train on {len(overlap)} keys"
                             f" — abort")
        assert len({r["emb"] for r in rows}) == len(rows), f"duplicate keys in '{g}'"

    chosen = [r for rows in packs.values() for r in rows]
    print(f"[pack-val] chose {len(chosen)} rows across {len(packs)} packs")
    for g, rows in sorted(packs.items()):
        bb = Counter(bucket_of(int(index[r["emb"]][2])) for r in rows)
        deficit = g_q.get(g, 0) - len(rows)
        print(f"[pack-val]   {g:8s} {len(rows):4d}/{g_q.get(g, 0):4d} rows"
              f"{f'  (short {deficit})' if deficit > 0 else ''}"
              f"  buckets {dict(sorted(bb.items()))}"
              f"  shards {len({index[r['emb']][0] for r in rows})}")

    if args.dry_run:
        print("[pack-val] dry run — nothing written")
        return 0

    cache_dir = tempfile.mkdtemp(prefix="valpack_")
    for g, rows in sorted(packs.items()):
        if not rows:
            continue
        _write_pack(args, g, rows, index, cache_dir)
    return 0


def _write_pack(args, group, chosen, index, cache_dir):
    """Write one group's parquet plus its source_key -> emb map."""
    import pyarrow.parquet as pq

    from vision_adapter.data.pack import pack_rows
    from vision_adapter.data.stream import _download_shard_hf_transfer

    by_shard: dict[str, dict] = defaultdict(dict)
    for r in chosen:
        sf, row = index[r["emb"]][0], index[r["emb"]][1]
        by_shard[sf][row] = r

    out_rows = []
    # the parquet's `key` is the SOURCE key, which is not the manifest's `emb`
    # — they only meet through the key index. This table is how the trainer
    # maps one to the other without a second index lookup.
    source_to_emb: dict[str, str] = {}
    for sf in sorted(by_shard):
        local = _download_shard_hf_transfer(sf, cache_dir=cache_dir)
        if not local:
            print(f"[pack-val] could not fetch {sf} — its rows are dropped")
            continue
        pf = pq.ParquetFile(local)
        wanted = by_shard[sf]
        n_done = 0
        for rgi in range(pf.num_row_groups):
            if n_done >= len(wanted):
                break
            tbl = pf.read_row_group(rgi, columns=["key", "n_vis", "vis_bytes"])
            keys = tbl.column("key").to_pylist()
            nvs = tbl.column("n_vis").to_pylist()
            vbs = tbl.column("vis_bytes").to_pylist()
            for j, k in enumerate(keys):
                r = wanted.get(j)
                if r is None:
                    continue
                grid = r.get("grid_thw")
                want = int(grid[0]) * int(grid[1]) * int(grid[2]) // 4 if grid else int(nvs[j])
                if int(nvs[j]) != want:
                    print(f"[pack-val] {k}: n_vis={nvs[j]} but grid says {want} "
                          f"— keeping the stored n_vis")
                out_rows.append({"key": k, "n_vis": int(nvs[j]),
                                 "vis_bytes": bytes(vbs[j])})
                source_to_emb[k] = r["emb"]
                n_done += 1
            del tbl, vbs
        print(f"[pack-val]   {sf}: {n_done}/{len(wanted)} rows", flush=True)

    out = Path(args.out_dir) / f"val_pack_{group}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pack_rows(out_rows, str(out), batch_size=64,
              progress=lambda n: print(f"[pack-val]   {group}: wrote {n}", flush=True))
    size_gb = out.stat().st_size / 2**30
    print(f"[pack-val] wrote {out} ({len(out_rows)} rows, {size_gb:.2f} GiB)")

    side = out.with_suffix(".map.json")
    side.write_text(json.dumps(source_to_emb, indent=1, sort_keys=True))
    print(f"[pack-val] {group} source_key -> emb map: {side} "
          f"({len(source_to_emb)} entries)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="emb_cache",
                    help="one val_pack_<group>.parquet is written per group")
    ap.add_argument("--n", type=int, default=1000,
                    help="total rows, split across groups by corpus proportion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key-index", default=None)
    ap.add_argument("--manifest", default=None, help="val manifest; HF when absent")
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the quotas and the disjoint check, write nothing")
    return pack(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())