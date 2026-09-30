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

## Audit manifest live (2026-09-30, lecture-seule, aucun fix implémenté)

- Les 52 924 rows `agentic` (100 %) portent un bloc image textuel Sero
  (`|begin_of_image| + 128×|image| + |end_of_image|`, compte fixe) = 265 tokens
  Qwen par row ; `make_collate` ne le strippe pas et `embeds_for` injecte les
  embeddings en plus → double représentation avérée, ~265/283 tokens du user
  médian agentic = markers.
- 38,6 % des rows train répètent une clé `emb` déjà vue (8 653 clés, tout en
  `doc`/`conv`, 0 en agentic) ; 47 % des rows val (1 128/2 400) ont leur `emb`
  dans train → val_loss `modal_train.py` optimiste.
- Comptes live : train 117 600 (agentic 52 924 / doc 52 908 / conv 11 768),
  val 2 400 — ni 120k ni 114 024. Détail chiffré : `qwen-injection-evidence.md`
  § Audit ; composition acceptée telle quelle (décision 2026-09-30).
