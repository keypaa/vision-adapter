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

## NEXT-2. Refactor `embeds_for` vers le protocole natif (local, TDD)
Placeholders + `mm_token_type_ids` + positions via `compute_3d_position_ids`
(ou équivalent), test d’équivalence loss vs ancien path sur modèle tiny CPU.
U1 doit être levé avant (règle d’expansion lue dans le processor).

## NEXT-3. Held-out stratifié + permutation (1 session Molab)
Rerun `eval_heldout.py --n 200` + patch groupement (10 lignes) : par `g`,
par bucket `n_vis`, par longueur de réponse + contrôle `vis` permutés.
Tranche U2 : vision vs shortcut.

## NEXT-4. Micro-bench unifié transfer + attention (1 session Molab)
`hf_transfer` vs Range (MiB/s, retries) + apportionnement par type de couche
sur backbone chargé une fois. Kill criteria NEXT_STEPS §3 avant tout code Flex.
