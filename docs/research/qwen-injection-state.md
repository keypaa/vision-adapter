# Qwen3.5 injection — STATE

Date : 2026-09-21. Statut : spec verrouillé sur sources, non implémenté.

## Ce qui est vrai

- Le protocole d’injection attendu par les créateurs (transformers `Qwen3_5Model.forward`,
  docs `model_doc/qwen3_5`) est : placeholders `image_token_id=248056` encadrés de
  `vision_start/end` (248053/248054) + `mm_token_type_ids` (0 texte / 1 image) +
  positions mRoPE 3D (`compute_3d_position_ids`) + scatter natif
  (`inputs_embeds.masked_scatter(placeholder_mask, image_embeds)`).
- Notre `embeds_for` (`vision_adapter/core.py:127-144`) rate ces 4 points : ids bruts
  sans placeholders, pas de `mm_token_type_ids`, RoPE texte sur le span visuel,
  splice à position fixe `[1:1+n_vis]`.
- `GROK_PROBE.md:16` mentionnait déjà le sentinel 248056 sans jamais l’exercer.
- Référence DeepSeek lue (mise de côté) : `DeepSeek-V4-Flash-Vision-Exp`,
  `encoding/README.md` (token inline `<｜deepseek_image｜>`, `media` en ordre) +
  `inference/README.md` (Vision + Aligner de référence). Cible = V4-Flash-0731,
  pas V4.1 ni Vision-Exp.

Voir aussi : `qwen-injection-evidence.md`, `qwen-injection-unknowns.md`,
`qwen-injection-risks.md`, `qwen-injection-next.md`.
