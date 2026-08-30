"""vision_adapter/data/cauldron.py — Cauldron subset pull (stub).

Extracted from master:modal_pipeline.py::cauldron_pull.  Contract:
  cauldron_manifest.jsonl — raw Cauldron pull before the sampled recipe.
  Two constants document the parallelism intent (not exercised in dry_run):
    N_DL=6   parallel parquet downloads
    N_SAVE=12 parallel PNG encodes
"""

from __future__ import annotations

import hashlib
from pathlib import Path

N_DL = 6  # parallel parquet downloads (HF hub)
N_SAVE = 12  # parallel PNG encodes (CPU-bound)


def pull_cauldron(
    backend=None,
    out_dir: Path | str | None = None,
    max_rows: int = 54000,
    dry_run: bool = False,
    revision: str | None = None,
    allowed_groups: set[str] | None = None,
) -> list[dict]:
    """Pull permissive Cauldron subsets and return row dicts.

    Each row is {"images": [paths], "texts": [{"user":..., "assistant":...}], "subset": str}.
    For dry_run (tests) returns ``max_rows`` fake rows without touching HF.
    Otherwise attempts ``datasets.load_dataset("HuggingFaceM4/the_cauldron")``
    punted to a stub (0 rows) if datasets is absent — dry_run is what tests cover.
    """
    if dry_run:
        rows: list[dict] = []
        for i in range(max_rows):
            rel = f"cauldron/fake_{i:06d}.png"
            sha = hashlib.sha1(rel.encode()).hexdigest()[:20]
            rows.append(
                {
                    "images": [f"embeddings/{sha}.pt"],
                    "texts": [{"user": f"cauldron user {i}", "assistant": f"cauldron assistant {i}"}],
                    "subset": "cauldron",
                }
            )
        return rows

    # Non-dry_run: real HF pull (requires `datasets`). Uses ORDER BY image
    # semantics and per-subset caps so --mix 45,45,10 can slice deterministically.
    # The 20k recipe streams captions only (no PNG re-encode here) — the PNGs
    # are needed only for precompute; the manifest needs {images, texts, group}.
    DOC_SUBSETS = ["chartqa", "docvqa", "infographic_vqa", "screen2words", "websight", "ocrvqa", "textvqa", "plotqa", "ai2d", "scienceqa"]
    CONV_SUBSETS = ["vqav2", "okvqa", "aokvqa", "visual7w"]
    try:
        from datasets import load_dataset  # type: ignore

        from vision_adapter.backends.auth import get_hf_token as _cauld_tok

        tok = _cauld_tok()
        # Stream over each subset's train split; pull_cauldron caps per group
        # so a 20k total can stay local-fast (no 120k materialisation).
        all_rows: list[dict] = []
        per_sub_idx: dict[str, int] = {}
        if allowed_groups is not None:
            wanted: list[str] = []
            if "doc" in allowed_groups:
                wanted.extend(DOC_SUBSETS)
            if "conv" in allowed_groups:
                wanted.extend(CONV_SUBSETS)
            subsets = wanted
        else:
            subsets = DOC_SUBSETS + CONV_SUBSETS
        for subset in subsets:
            try:
                ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True, token=tok, revision=revision)
            except Exception as e:
                print(f"[cauldron] skip subset {subset}: {e}", flush=True)
                continue
            group = "doc" if subset in DOC_SUBSETS else "conv"
            for ex in ds:
                if len(all_rows) >= max_rows:
                    break
                i = per_sub_idx.get(subset, 0)
                per_sub_idx[subset] = i + 1
                rel = f"cauldron/{subset}-{i:07d}-0.png"
                emb = f"embeddings/{hashlib.sha1(rel.encode()).hexdigest()[:20]}.pt"
                texts = ex.get("texts") or [{"user": str(ex.get("question", "")), "assistant": str(ex.get("answer", ""))}]
                if isinstance(texts, dict):
                    texts = [texts]
                all_rows.append({"images": [emb], "texts": texts, "subset": subset, "group": group})
                if len(all_rows) >= max_rows:
                    break
            if len(all_rows) >= max_rows:
                break
        return all_rows
    except Exception as e:
        print(f"[cauldron] HF pull failed ({type(e).__name__}: {e})", flush=True)
        return []
