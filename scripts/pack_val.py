#!/usr/bin/env python3
"""Pack the val split into ONE parquet, so a probe costs one fetch not 21.

The disjoint manifest (`train_manifest_val_disjoint.jsonl`) already guarantees
that no val row's `emb` key appears in the train manifest — verified by an
assert when it was built (scripts/rebuild_val_disjoint.py). So there is no
disjointness work to do here. What is broken is the *plan*: run_train rebuilds
the val plan with `excluded_shards=set(plan.keys())`, which drops 1272 rows to
77 spread over 21 shards. Streaming 21 shards for 77 rows costs ~10 min per
probe against ~2 min of training between probes.

This packs ~N rows, chosen to reproduce the corpus's g mix and n_vis bucket
mix, into a single parquet. One fetch, ~1000 rows, every domain represented.

RAM-bounded: reads one row group at a time, two columns at a time.

Usage:
    python scripts/pack_val.py --out emb_cache/val_pack.parquet --n 1000
    python scripts/pack_val.py --out ... --n 1000 --dry-run   # quotas only
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The six buckets from the dataset card, and the 45/45/10 g mix.
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


def pick_rows(val_rows, index, g_q, b_q, seed=0):
    """One row per (g, bucket) cell until the quota is met.

    Cells are filled in order of scarcity (fewest candidates first) so a rare
    bucket is not starved by a common one that happens to be enumerated first.

    The live val manifest repeats `emb` keys across rows (measured 2026-10-04:
    1272 rows, fewer unique keys). Deduplicate first — two rows sharing an
    embedding are the same val sample twice, which would make the loss look
    steadier than it is.
    """
    import random

    rng = random.Random(seed)
    seen_keys = set()
    cells: dict[tuple, list] = defaultdict(list)
    for r in val_rows:
        e = r.get("emb")
        if not e or e in seen_keys:
            continue
        loc = index.get(e)
        if not loc or len(loc) != 3:
            continue
        seen_keys.add(e)
        cells[(r.get("g", "?"), bucket_of(int(loc[2])))].append(r)

    chosen, deficit = [], 0
    order = sorted(cells, key=lambda c: (len(cells[c]), c))
    for cell in order:
        want = min(g_q.get(cell[0], 0), b_q.get(cell[1], 0))
        pool = cells[cell]
        rng.shuffle(pool)
        chosen.extend(pool[:want])

    # quotas rarely divide evenly; top up from whatever is left, rarest first
    if len(chosen) < sum(g_q.values()):
        picked = {id(r) for r in chosen}
        rest = [r for cell in order for r in cells[cell] if id(r) not in picked]
        rng.shuffle(rest)
        chosen.extend(rest[: sum(g_q.values()) - len(chosen)])

    rng.shuffle(chosen)
    deficit = sum(g_q.values()) - len(chosen)
    return chosen, deficit


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
    chosen, deficit = pick_rows(val_rows, index, g_q, b_q, seed=args.seed)

    # the invariant the whole script exists to preserve
    train_embs = {r["emb"] for r in train_rows if r.get("emb")}
    overlap = [r["emb"] for r in chosen if r["emb"] in train_embs]
    assert not overlap, f"val pack overlaps train on {len(overlap)} keys — abort"
    assert len({r["emb"] for r in chosen}) == len(chosen), "duplicate keys in the pack"

    gb = Counter(r.get("g", "?") for r in chosen)
    bb = Counter(bucket_of(int(index[r["emb"]][2])) for r in chosen)
    print(f"[pack-val] chose {len(chosen)} rows (deficit {deficit})")
    print(f"[pack-val]   g      {dict(gb)}   target {g_q}")
    print(f"[pack-val]   bucket {dict(sorted(bb.items()))}   target {b_q}")
    print(f"[pack-val]   shards {len({index[r['emb']][0] for r in chosen})} distinct")

    if args.dry_run:
        print("[pack-val] dry run — nothing written")
        return 0

    by_shard: dict[str, dict] = defaultdict(dict)
    for r in chosen:
        sf, row = index[r["emb"]][0], index[r["emb"]][1]
        by_shard[sf][row] = r

    cache_dir = tempfile.mkdtemp(prefix="valpack_")
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pack_rows(out_rows, str(out), batch_size=64,
              progress=lambda n: print(f"[pack-val]   wrote {n}", flush=True))
    size_gb = out.stat().st_size / 2**30
    print(f"[pack-val] wrote {out} ({len(out_rows)} rows, {size_gb:.2f} GiB)")

    side = out.with_suffix(".map.json")
    side.write_text(json.dumps(source_to_emb, indent=1, sort_keys=True))
    print(f"[pack-val] source_key -> emb map: {side} ({len(source_to_emb)} entries)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="emb_cache/val_pack.parquet")
    ap.add_argument("--n", type=int, default=1000, help="rows to pack")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key-index", default=None)
    ap.add_argument("--manifest", default=None, help="val manifest; HF when absent")
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the quotas and the disjoint check, write nothing")
    return pack(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())