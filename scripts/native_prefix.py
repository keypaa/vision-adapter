"""Eval-only native-protocol prefix builder (Qwen3.5, U1 expansion rule).

Builds the exact ``Qwen3_5Model.forward`` generation-prefix inputs from a
training-collated batch WITHOUT touching the training path
(``vision_adapter/train.py``, ``core.embeds_for`` — read-only here; the
``test_training_path_untouched`` test pins that).

Expansion rule (verified against transformers 5.12.1 source, local
site-packages — never from memory):

- placeholders/image ``N = prod(grid_thw) // merge_size**2``
  (``Qwen3VLProcessor.replace_image_token``:
  ``image_grid_thw.prod() // merge_size**2`` with
  ``merge_size = image_processor.merge_size = 2``; identical formula in
  ``Qwen3VLModel.get_image_features``:
  ``split_sizes = image_grid_thw.prod(-1) // spatial_merge_size**2``,
  ``vision_config.spatial_merge_size = 2``).
- framing: ``<vision_start>`` immediately before the first ``<image_pad>``,
  ``<vision_end>`` immediately after the last one. The model counts an
  image only where vision_start directly precedes an image pad
  (``_get_image_nums_and_video_nums``); the video path builds
  ``vision_start + pads + vision_end`` per frame. For images the prompt
  side supplies the frame (``replace_image_token`` returns pads only).
- ``mm_token_type_ids``: 0 text / 1 image / 2 video
  (``create_mm_token_type_ids``; the processor's ``image_token_ids``
  contains ONLY ``image_token_id``, so vision_start/end stay 0).
- ``get_placeholder_mask`` requires ``#image_token_id == len(image_embeds)``,
  hence this builder asserts ``N == n_vis`` per row (cached vis length must
  match the grid-derived placeholder count).
- ``position_ids`` mirrors ``get_rope_index``/``compute_3d_position_ids``:
  contiguous same-type spans grouped by ``mm_token_type_ids``; text spans
  advance ``current_pos`` by their length; each vision span consumes the
  next grid, assigns ``get_vision_position_ids(current_pos, grid, 1,
  spatial_merge_size)``, then
  ``current_pos += max(h, w) // spatial_merge_size``. ``compute_3d``
  raises when multimodal grids arrive without ``mm_token_type_ids`` —
  this builder always returns them together.

Pure shape contract
-------------------
Input ``batch`` (from ``vision_adapter.core.make_collate``):
``[BOS?][img x n_vis][user][answer][EOS]`` + right pad; ``labels`` are
``-100`` except over ``answer+EOS``.

For each row ``i`` with grid ``(t, h, w)`` and ``N = t*h*w // merge**2``:

- ``cut`` = first supervised index (``labels != -100``) if labels exist,
  else first EOS hit, else the attention-mask length. The prefix is
  ``input_ids[i, :cut]`` (BOS slot + raw image span + user tokens —
  answer and EOS excluded, so a generation prefix never ends on EOS).
- the raw image span ``[1:1+n_vis]`` (left as pad by the collate) is
  replaced by ``[vision_start] + [image_token]*N + [vision_end]``.
- rows are right-padded with ``pad_id`` to the batch max; ``attention_mask``
  is 1 over live tokens, 0 over pad; ``mm_token_type_ids`` is 1 exactly
  where ``input_ids == image_token_id``, else 0.

Returns ``dict`` with ``input_ids`` (B, L'), ``attention_mask`` (B, L'),
``mm_token_type_ids`` (B, L'), ``position_ids`` (3, B, L') and
``image_grid_thw`` (B, 3) — the exact inputs ``Qwen3_5Model.forward``
needs for a native-protocol generation prefix (NEXT-1). Deterministic:
pure tensor ops, no randomness.
"""
from __future__ import annotations

import torch

IMAGE_TOKEN_ID = 248056
VISION_START_ID = 248053
VISION_END_ID = 248054


def placeholder_count(grid_thw_row: torch.Tensor, merge_size: int = 2) -> int:
    """Placeholders for one image: ``prod(grid_thw) // merge_size**2``."""
    return int(torch.prod(torch.as_tensor(grid_thw_row)).item()) // (merge_size**2)


def _vision_position_ids(
    start: int, grid_thw_row: torch.Tensor, spatial_merge_size: int = 2
) -> torch.Tensor:
    """Mirror of ``Qwen3VLModel.get_vision_position_ids`` (temp_merge=1).

    Returns ``(3, N)`` long tensor of (temporal, height, width) indices with
    the source's repeat pattern (width fastest, then height, then temporal).
    """
    t, h, w = (int(v) for v in torch.as_tensor(grid_thw_row).tolist())
    llm_t, llm_h, llm_w = t, h // spatial_merge_size, w // spatial_merge_size
    pos_w = torch.arange(llm_w) + start
    pos_h = torch.arange(llm_h) + start
    pos_t = torch.arange(llm_t)  # time_interval=1; start added after repeat
    pos_w = pos_w.repeat(llm_h * llm_t)
    pos_h = pos_h.repeat_interleave(llm_w).repeat(llm_t)
    pos_t = pos_t.repeat_interleave(llm_h * llm_w) + start
    return torch.stack([pos_t, pos_h, pos_w], dim=0).long()


def build_native_prefix(
    batch: dict,
    tokenizer,
    grid_thw: torch.Tensor,
    merge_size: int = 2,
) -> dict:
    """Build native-protocol generation-prefix inputs from a collated batch.

    Args:
        batch: ``make_collate`` output with ``input_ids``, ``attention_mask``,
            ``n_vis`` and optionally ``labels``.
        tokenizer: provides ``image_token_id`` / ``vision_start_token_id`` /
            ``vision_end_token_id`` / ``eos_token_id`` / ``pad_token_id``
            (sensible Qwen3.5 defaults when attributes are missing).
        grid_thw: ``(B, 3)`` long tensor, one ``(t, h, w)`` LLM-grid row per
            batch item (one image per row in the collate layout).
        merge_size: spatial merge factor (default 2, the vision-config value).

    Raises:
        ValueError: when a grid is not divisible by ``merge_size**2`` or its
            placeholder count differs from the row's ``n_vis`` (the scatter
            contract ``#pads == len(image_embeds)`` would break downstream).
    """
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

    counts = [placeholder_count(grids[i], merge_size) for i in range(B)]
    for i in range(B):
        if int(torch.prod(grids[i]).item()) % (merge_size**2) != 0:
            raise ValueError(f"row {i}: grid {grids[i].tolist()} not divisible by merge**2={merge_size**2}")
        if counts[i] != n_vis[i]:
            raise ValueError(
                f"row {i}: grid gives N={counts[i]} placeholders but n_vis={n_vis[i]} "
                "(native scatter needs #pads == len(image_embeds))"
            )

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

    # 3D mRoPE positions mirroring get_rope_index (grouped by mm type,
    # computed over the live span only; pad columns stay 0).
    out_pos = torch.zeros((3, B, L), dtype=torch.long)
    for i in range(B):
        live = int(out_attn[i].sum().item())
        types = out_mm[i, :live].tolist()
        groups: list[tuple[int, int, int]] = []
        p = 0
        while p < live:
            q = p
            while q < live and types[q] == types[p]:
                q += 1
            groups.append((types[p], p, q))
            p = q
        grid_iter = iter([grids[i]])
        cols: list[torch.Tensor] = []
        cur = 0
        for mtype, s, e in groups:
            if mtype == 0:
                cols.append(torch.arange(e - s).view(1, -1).expand(3, -1) + cur)
                cur += e - s
            else:
                g = next(grid_iter)
                cols.append(_vision_position_ids(cur, g, merge_size))
                cur += max(int(g[1].item()), int(g[2].item())) // merge_size
        if cols:
            span = torch.cat(cols, dim=1)
            out_pos[:, i, :live] = span

    return {
        "input_ids": out_ids,
        "attention_mask": out_attn,
        "mm_token_type_ids": out_mm,
        "position_ids": out_pos,
        "image_grid_thw": grids.clone(),
    }
