"""Native-protocol batch layouts (Qwen3.5 creator protocol), single owner.

Used by train (full-length + labels), eval generation (prefix) and eval
loss paths. ``scripts/native_prefix.py`` only re-exports from here.

Creator protocol per row: ``[slot][vstart][pads x N][vend][user...]``
with ``N == n_vis`` asserted, ``mm=1`` on pads only, mRoPE positions with
native post-vision offset. Grids are synthetic (``grid_for_nvis``): the
precomputed parquet stores no MoonViT geometry (documented approximation).
"""
from __future__ import annotations

import torch

from vision_adapter.core import grid_for_nvis, placeholder_count, vision_position_ids

IMAGE_TOKEN_ID = 248056
VISION_START_ID = 248053
VISION_END_ID = 248054


def _check_grids(grids: torch.Tensor, n_vis: list[int], merge_size: int, B: int) -> list[int]:
    counts = [placeholder_count(grids[i], merge_size) for i in range(B)]
    for i in range(B):
        if int(torch.prod(grids[i]).item()) % (merge_size**2) != 0:
            raise ValueError(f"row {i}: grid {grids[i].tolist()} not divisible by merge**2={merge_size**2}")
        if counts[i] != n_vis[i]:
            raise ValueError(
                f"row {i}: grid gives N={counts[i]} placeholders but n_vis={n_vis[i]} "
                "(native scatter needs #pads == len(image_embeds))"
            )
    return counts


def _grouped_positions(types: list[int], live: int, grid: torch.Tensor, merge_size: int) -> torch.Tensor:
    """mRoPE columns over one live span, native cursor semantics (text
    advances, vision consumes grid then jumps max(h,w)//merge)."""
    groups: list[tuple[int, int, int]] = []
    p = 0
    while p < live:
        q = p
        while q < live and types[q] == types[p]:
            q += 1
        groups.append((types[p], p, q))
        p = q
    cols: list[torch.Tensor] = []
    cur = 0
    used_grid = False
    for mtype, s, e in groups:
        if mtype == 0:
            cols.append(torch.arange(e - s).view(1, -1).expand(3, -1) + cur)
            cur += e - s
        else:
            if used_grid:
                raise ValueError("multiple vision spans per row are not supported")
            used_grid = True
            cols.append(vision_position_ids(cur, grid, merge_size))
            cur += max(int(grid[1].item()), int(grid[2].item())) // merge_size
    return torch.cat(cols, dim=1) if cols else torch.zeros((3, 0), dtype=torch.long)


def build_native_prefix(batch: dict, tokenizer, grid_thw: torch.Tensor, merge_size: int = 2) -> dict:
    """Native-protocol generation-prefix inputs (answer cut). See module doc."""
    image_id = int(getattr(tokenizer, "image_token_id", IMAGE_TOKEN_ID))
    vstart_id = int(getattr(tokenizer, "vision_start_token_id", VISION_START_ID))
    vend_id = int(getattr(tokenizer, "vision_end_token_id", VISION_END_ID))
    eos_id = int(getattr(tokenizer, "eos_token_id", 2))
    pad_id = int(getattr(tokenizer, "pad_token_id", 0))

    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    n_vis = [int(n) for n in batch["n_vis"].tolist()]
    labels = batch.get("labels")
    grids = torch.as_tensor(grid_thw, dtype=torch.long)
    B = ids.shape[0]
    if grids.ndim != 2 or grids.shape[0] != B or grids.shape[1] != 3:
        raise ValueError(f"grid_thw must be (B, 3) with B={B}, got {tuple(grids.shape)}")
    counts = _check_grids(grids, n_vis, merge_size, B)

    new_rows: list[list[int]] = []
    for i in range(B):
        row = ids[i].tolist()
        mask_len = int(attn[i].sum().item())
        if labels is not None:
            lab = labels[i].tolist()
            sup = [p for p, v in enumerate(lab) if v != -100]
            cut = sup[0] if sup else mask_len
        else:
            hits = [p for p in range(mask_len) if row[p] == eos_id]
            cut = hits[0] if hits else mask_len
        cut = max(0, min(cut, mask_len))
        prefix = row[:cut]
        nv = n_vis[i]
        framed = [vstart_id] + [image_id] * counts[i] + [vend_id]
        new_rows.append(prefix[:1] + framed + prefix[1 + nv:])

    L = max(len(r) for r in new_rows)
    out_ids = torch.full((B, L), pad_id, dtype=torch.long)
    out_attn = torch.zeros((B, L), dtype=torch.long)
    out_mm = torch.zeros((B, L), dtype=torch.long)
    for i, r in enumerate(new_rows):
        out_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        out_attn[i, : len(r)] = 1
        out_mm[i, : len(r)] = torch.tensor([1 if t == image_id else 0 for t in r], dtype=torch.long)

    out_pos = torch.zeros((3, B, L), dtype=torch.long)
    for i in range(B):
        live = int(out_attn[i].sum().item())
        out_pos[:, i, :live] = _grouped_positions(out_mm[i, :live].tolist(), live, grids[i], merge_size)

    return {
        "input_ids": out_ids,
        "attention_mask": out_attn,
        "mm_token_type_ids": out_mm,
        "position_ids": out_pos,
        "image_grid_thw": grids.clone(),
    }


def build_native_train_batch(
    batch: dict,
    grid_thw: torch.Tensor | None = None,
    *,
    image_token_id: int | None = None,
    vision_start_id: int | None = None,
    vision_end_id: int | None = None,
    pad_token_id: int = 0,
    merge_size: int = 2,
) -> dict:
    """Full-length native layout for training (labels shifted, grad-ready).

    Same framing as the prefix builder but keeps answer+EOS with labels
    shifted by +(2+N-nv) (== +2, N==nv asserted). Returns 3-row native
    mRoPE positions (text decoder prepends its arange row); see
    :func:`native_train_forward` for the 4-row assembly.
    ``grid_thw=None`` derives synthetic grids from ``n_vis``.
    """
    if image_token_id is None:
        image_token_id = IMAGE_TOKEN_ID
    if vision_start_id is None:
        vision_start_id = VISION_START_ID
    if vision_end_id is None:
        vision_end_id = VISION_END_ID
    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = batch["labels"]
    n_vis = [int(n) for n in batch["n_vis"].tolist()]
    B, L = ids.shape[:2]
    if grid_thw is None:
        grids = torch.stack([grid_for_nvis(nv, merge_size) for nv in n_vis])
    else:
        grids = torch.as_tensor(grid_thw, dtype=torch.long)
        if grids.ndim != 2 or grids.shape[0] != B or grids.shape[1] != 3:
            raise ValueError(f"grid_thw must be (B, 3) with B={B}, got {tuple(grids.shape)}")
    counts = _check_grids(grids, n_vis, merge_size, B)

    new_rows: list[list[int]] = []
    new_labs: list[list[int]] = []
    for i in range(B):
        row = ids[i].tolist()
        lab = labels[i].tolist()
        mask_len = int(attn[i].sum().item())
        nv = n_vis[i]
        if nv == 0:
            new_rows.append(row[:mask_len])
            new_labs.append(lab[:mask_len])
            continue
        delta = (2 + counts[i]) - nv
        framed = [vision_start_id] + [image_token_id] * counts[i] + [vision_end_id]
        new_rows.append(row[:1] + framed + row[1 + nv: mask_len])
        shifted = [-100] * (mask_len + delta)
        for p, v in enumerate(lab[:mask_len]):
            if v != -100:
                shifted[p + delta] = v
        new_labs.append(shifted)

    Lp = max(len(r) for r in new_rows)
    out_ids = torch.full((B, Lp), pad_token_id, dtype=torch.long)
    out_attn = torch.zeros((B, Lp), dtype=torch.long)
    out_lab = torch.full((B, Lp), -100, dtype=torch.long)
    out_mm = torch.zeros((B, Lp), dtype=torch.long)
    for i in range(B):
        r = new_rows[i]
        out_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        out_attn[i, : len(r)] = 1
        out_lab[i, : len(new_labs[i])] = torch.tensor(new_labs[i], dtype=torch.long)
        out_mm[i, : len(r)] = torch.tensor([1 if t == image_token_id else 0 for t in r], dtype=torch.long)

    out_pos = torch.zeros((3, B, Lp), dtype=torch.long)
    for i in range(B):
        live = int(out_attn[i].sum().item())
        out_pos[:, i, :live] = _grouped_positions(out_mm[i, :live].tolist(), live, grids[i], merge_size)

    return {
        "input_ids": out_ids,
        "attention_mask": out_attn,
        "labels": out_lab,
        "mm_token_type_ids": out_mm,
        "position_ids": out_pos,
        "image_grid_thw": grids.clone(),
    }


def native_train_forward(model, proj, batch: dict, device: str, merge_size: int = 2):
    """Full native forward inputs for the training path (grad flows to proj).

    Returns ``(merged_embeds, labels, attention_mask, position_ids_4row)``
    with ``position_ids`` row 0 plain arange (text default) and rows 1-3
    native mRoPE. Keeps ``train_step_qwen`` under the complexity budget.
    """
    nat = build_native_train_batch(batch, None, merge_size=merge_size)
    table = model.get_input_embeddings()
    out_dtype = next(model.parameters()).dtype
    proj_dtype = next(proj.parameters()).dtype
    base = table(nat["input_ids"].to(device)).to(out_dtype)
    pv = proj(batch["vis"].to(device).to(proj_dtype)).to(out_dtype)
    merged = base.clone()
    n_vis = [int(n) for n in batch["n_vis"].tolist()]
    mask = nat["input_ids"] == IMAGE_TOKEN_ID
    merged = scatter_projector_outputs(merged, pv, mask, n_vis)
    B, L = nat["input_ids"].shape[:2]
    pos4 = torch.zeros((4, B, L), dtype=torch.long)
    pos4[0] = torch.arange(L).view(1, -1).expand(B, -1)
    pos4[1:] = nat["position_ids"]
    return merged, nat["labels"], nat["attention_mask"], pos4.to(device)


def scatter_projector_outputs(merged: torch.Tensor, proj_out: torch.Tensor,
                              placeholder_mask: torch.Tensor, n_vis: list[int]) -> torch.Tensor:
    """In-place scatter of projector rows onto placeholder positions (grad-safe).

    ``merged`` is modified and returned. Row ``i`` takes ``proj_out[i, :nv]``
    at its mask positions (``#pads == nv`` asserted by the builders).
    """
    for i, nv in enumerate(n_vis):
        hits = placeholder_mask[i].nonzero(as_tuple=False).squeeze(-1)
        assert int(hits.numel()) == nv, f"row {i}: {int(hits.numel())} pads != n_vis {nv}"
        merged[i, hits] = proj_out[i, :nv]
    return merged
