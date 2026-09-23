# Qwen3.5 injection — NEXT

Ordre imposé (chacun conditionne le suivant, aucun training > 1h).

## NEXT-1. Sanity génération native (~15 min Molab, 0 training)
`scripts/colab_unsloth_test.py --prefix-mode native` : grille synthétique
`grid_for_nvis` (N==n_vis), `build_native_generate_inputs` (scatter projecteur
au masque + mm + mRoPE), sampling carte VL. Contrat CPU pinné
(`tests/test_native_prefix.py`, 120 tests verts) ; continuation cache à valider
GPU (kwargs `mm_token_type_ids`/`image_grid_thw`/`position_ids` passés à `generate`).
- Si ça génère : spec bon → NEXT-2.
- Si vide aussi : cause = frontière `generate` (MTP/cache) → fermer la
  piste injection, documenter, passer au serving stack si besoin.

## NEXT-2. Training protocole natif — VERDICT NÉGATIF (2026-09-23, Molab)
Différentiel 2×300 steps (même tête scalée, mêmes batches) + held-out
60 rows, chaque ckpt dans son régime : C-splice 0.894 vs N-natif 0.950
(+5,9 % splice, gate 10 % manquée) ; trajectoires : C 0.845/0.970 vs
N 0.885/1.020, même sens. Le layout natif complet n’apporte rien à 300 steps.
DÉCISION : on garde splice+mRoPE par défaut, piste natif-training classée
sans suppression de code (réversible si nouvelles données). Le module
`vision_adapter/native.py` reste utile au harness de génération.

## NEXT-3. Held-out stratifié + permutation (1 session Molab)
Rerun `eval_heldout.py --n 200` + patch groupement (10 lignes) : par `g`,
par bucket `n_vis`, par longueur de réponse + contrôle `vis` permutés.
Tranche U2 : vision vs shortcut.

## NEXT-5. Tête normalisée + diag court (1 session Molab ≤ 1h)
Variante A : LayerNorm finale × RMS table mesuré (cible ~0.02, constante
documentée) — déterministe, pas d’échelle ré-apprenable qui re-explose.
Diag 200-400 steps pires shards + gate : normes saines (rms visuel ≈ table
×0.5-2), held-out vs baseline, génération non-vide. Si vert → refactor
d’architecture acté (MODE E-lite) avant tout run long.

## NEXT-4. Micro-bench unifié transfer + attention (1 session Molab)
`hf_transfer` vs Range (MiB/s, retries) + apportionnement par type de couche
sur backbone chargé une fois. Kill criteria NEXT_STEPS §3 avant tout code Flex.
