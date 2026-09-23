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

## NEXT-2. Training protocole natif — IMPLÉMENTÉ, en attente du différentiel
`vision_adapter/native.py` (propriété unique) + `train_step_qwen` sous
`VISION_ADAPTER_NATIVE_TRAIN=1` (défaut 0, legacy inchangé) : placeholders +
framing + mm + offset natif post-vision, labels shiftés +2. 148 tests verts.
Différentiel prévu : natif vs courant, 300 steps chacun, même tête scalée,
même ordre, + held-out — gate : natif ≥ courant.

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
