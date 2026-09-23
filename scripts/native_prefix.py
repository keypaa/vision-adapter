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

from vision_adapter.core import grid_for_nvis as grid_for_nvis
from vision_adapter.core import placeholder_count as placeholder_count
from vision_adapter.native import IMAGE_TOKEN_ID, build_native_prefix


def build_native_generate_inputs(model, proj, batch: dict, tokenizer, grid_thw: torch.Tensor,
                                 device: str, merge_size: int = 2) -> dict:
    """Generate-ready native-protocol inputs with OUR projector outputs scattered.

    Replicates what ``Qwen3_5Model.forward`` does natively
    (``masked_scatter`` at the placeholder mask) but with our projector
    instead of the native vision encoder: embed-table base + projector
    outputs at ``input_ids == image_token_id`` positions, plus ``mm`` ids,
    ``image_grid_thw`` and mRoPE ``position_ids`` for the generate call.
    No-grad (eval). Cache continuation inside ``generate`` stays
    MOLAB-VERIFIED (NEXT-1) — CPU pins the contract, GPU proves behavior.
    """
    native = build_native_prefix(batch, tokenizer, grid_thw, merge_size)
    image_id = int(getattr(tokenizer, "image_token_id", IMAGE_TOKEN_ID))
    ids = native["input_ids"].to(device)
    attn = native["attention_mask"].to(device)
    mm = native["mm_token_type_ids"].to(device)
    pos = native["position_ids"].to(device)
    table = model.get_input_embeddings()
    out_dtype = next(model.parameters()).dtype
    proj_dtype = next(proj.parameters()).dtype
    with torch.no_grad():
        base = table(ids).to(out_dtype)
        vis = batch["vis"].to(device).to(proj_dtype)
        pv = proj(vis).to(out_dtype)
        merged = base.clone()
        n_vis = [int(n) for n in batch["n_vis"].tolist()]
        for i in range(ids.shape[0]):
            hits = (ids[i] == image_id).nonzero(as_tuple=False).squeeze(-1)
            nv = n_vis[i]
            assert int(hits.numel()) == nv, f"row {i}: {int(hits.numel())} pads != n_vis {nv}"
            merged[i, hits] = pv[i, :nv]
    return {
        "inputs_embeds": merged,
        "attention_mask": attn,
        "mm_token_type_ids": mm,
        "image_grid_thw": native["image_grid_thw"].to(device),
        "position_ids": pos,
    }

