# Qwen3.5 injection — EVIDENCE

## Sources créateurs (vérifiées par lecture, sept. 2026)

1. `src/transformers/models/qwen3_5/modular_qwen3_5.py`, classe `Qwen3_5Model.forward` :
   `masked_scatter` des `image_embeds` au masque des placeholders ;
   `compute_3d_position_ids(input_ids, image_grid_thw, ...)` pour le mRoPE ;
   `get_placeholder_mask` + `split_sizes = grid_thw.prod(-1) // merge²`.
2. `huggingface.co/docs/transformers/model_doc/qwen3_5` : usage via
   `Qwen3_5ForConditionalGeneration` + processor + `apply_chat_template` ;
   `image_token_id=248056`, `vision_start/end=248053/248054`,
   `mm_token_type_ids` (0/1/2) ; avertissement mRoPE (préserver le split
   temporel/hauteur/largeur sous peine de désalignement image).
3. `huggingface.co/Qwen/Qwen3.5-2B` (model card) : multimodal natif early-fusion ;
   génération via chat template + recipe sampling VL
   (temp 0.7, top_p 0.8, top_k 20).

## Sources praticiens (recoupées, pas parole d’évangile)

4. `kingbackyang/Megatron-LM-Qwen3_5`, `QWEN3_5_CODE_CHANGES_PR.md` :
   tour vision non-CLIP (Conv3d, pas de class token), sémantique
   `merger.norm` pré-norm avant pixel-shuffle, `image_token_lengths` +
   `image_grid_thw` portés pendant le training, 4 pièges documentés dont
   « pas juste encoder l’image en tokens » et mRoPE/decode à moitié justes.
5. `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, `encoding/README.md` et
   `inference/README.md` (lus en raw) : placeholder inline, aligner de référence.

## Preuves repo (locales)

- `vision_adapter/core.py:127-144` (`embeds_for`) vs `:151-197` (`visual_inject`).
- `tests/test_eval_heldout.py` + `scripts/eval_heldout.py` (held-out -33 %).
- `scripts/colab_unsloth_test.py` : `Gen: ''` ×2 ckpts tous régimes.
