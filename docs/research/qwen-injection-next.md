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

## NEXT-6. Suite audit externe — NOTÉ, NON IMPLÉMENTÉ (décision 2026-09-30)
L'audit 2026-09-30 a quantifié markers (265 tok/row), truncation (overflow
avéré ≥5k), dups (38,6 % rows), val overlap (47 %). Composition dataset
acceptée telle quelle. Fixes candidats, un à la fois, TDD + tests, quand le
terme échoit : (a) strip markers dans `make_collate` ; (b) garde overflow
`L` (skip/cap `n_vis>max_len`) ; (c) val disjoint par `emb` ou biais
documenté. (b) IMPLÉMENTÉ 2026-09-30 (TDD : `tests/test_collate_overflow.py`,
garde dans `vision_adapter/core.py:make_collate`, 159 tests verts, `ruff check`
ok). (a) IMPLÉMENTÉ 2026-09-30 (TDD : `tests/test_marker_strip.py`,
`strip_text_image_markers` + usage dans `make_collate`, 161 tests verts,
`ruff check` ok). (c) REBUILD PRÊT 2026-09-30 (TDD :
`tests/test_disjoint_val.py`, `scripts/rebuild_val_disjoint.py`, 164 tests
verts) : `train_manifest_val_disjoint.jsonl` (1 272 rows, overlap 0 vérifié,
mais 84,6 % agentic — val honnête, pas val représentative). Reste l'étape
humaine : push HF sous nouveau nom + switch explicite de
`modal_train.py:VAL_MANIFEST_REL` (commandes ci-dessous, jamais poussé par
l'agent : token d'écriture requis) :
`hf upload keypa/vision-adapter-manifests /tmp/opencode/train_manifest_val_disjoint.jsonl train_manifest_val_disjoint.jsonl --repo-type dataset`
— **exécuté 2026-09-30, les deux fichiers coexistent sur le repo**.

## NEXT-8. Val streaming branché (ÉTAPE 3, FAIT 2026-09-30)
`vision_adapter/train.py` (chemin des runs 20h) n'avait **aucune validation** :
`grep val` ne trouvait rien. Un run pouvait surapprendre sans le montrer.
`cfg.val_every=250` existait mais n'était câblé nulle part — champ mort.

FIX : plan val construit depuis `train_manifest_val_disjoint.jsonl` (le
disjoint, pas l'original à 47 % d'overlap), `excluded_shards=set(plan.keys())`
pour qu'aucun shard ne serve train et val, probe tous les `val_every` steps +
toujours au dernier step, record `type:"val"` distinct (pas de `gnorm`/`ema_loss`
donc jamais confondu avec une ligne train). La loss passe par `_batch_loss`,
qui **réutilise la recette exacte de `train_step_qwen`** (même shift, même
masque -100, même `lm_head`) : l'écart val/train mesure donc un vrai
surapprentissage, pas une différence de recette. Re-stream à chaque probe,
pas de cache local (disque/RAM contraints — décision 2026-09-30).
Un probe qui échoue log un WARN et n'arrête pas le run.
Pins : `tests/test_val_plan.py` (11 tests), 193 tests verts.

Reste pour cette étape : le val disjoint est sur HF mais `grid_thw` non.

## NEXT-9. NaN projector grads en fp16 — TROUVÉ ET FIXÉ 2026-09-30
Symptôme : premier test du run streaming sur T4 → `[train][WARN] non-finite
at 1, skipping`, aucune loss. Diagnostic stage par stage (forward/backward
séparés, données synthétiques) :

| Test | Résultat |
|---|---|
| forward fp16, 4100 tokens | ✅ hidden `absmax 48`, logits 10,7, loss 9,16 |
| **backward fp16** | ❌ **`gnorm=nan`, 6/6 params non-finite** |
| **backward bf16** | ✅ **`gnorm=522`, 0/6 non-finite** |

Le forward fp16 est sain : c'est le **backward à travers le backbone gelé**
qui sature. Le projector est le seul module fp32, et les valeurs qui
débordent sont des **activations** (plafond fp16 = 65 504), pas des grads de
paramètres fp16 — un `GradScaler` ne peut donc pas aider, il ne voit pas ces
grads. bf16 (plafond 3,4e38) supprime le mode de défaillance.

Blackwell aurait corrigé ça via `cc>=80 → bf16`, mais bf16 **marche aussi
sur T4** (cc 7.5) : aucune raison de payer un GPU plus cher.

FIX : `resolve_dtype(capability, dtype_arg)` dans `train.py` — `auto` ne
retourne **plus jamais** fp16, seulement bf16 (cc≥70) ou fp32 (cc<70, pas de
bf16). `fp16` explicite reste honoré comme échappatoire. Les deux sites de
sélection (local ligne 641, streaming ligne 775) **divergaient** (fp32 vs
fp16 sur T4) — ils partagent maintenant le même helper, c'est exactement cette
duplication qui avait caché le bug d'un côté. Pins :
`tests/test_dtype_selection.py` (10 tests), 203 tests verts.

## NEXT-7. Sidecar de géométrie sur le corpus complet (code prêt, données à produire)
Le grid synthétique inversait l'orientation (×3,0 d'aspect médian, 200/200
rows) — voir evidence. Le code est en place et testé (182 tests verts) :
`grid_from_dims`, `row_grids_for_batch`, `vision_adapter/grid_sidecar.py`,
`scripts/build_grid_sidecar.py`, `cfg.grid_sidecar`, `grid_source` dans la run
card. Reste à produire le sidecar sur les **138 987 embeddings** (le proof sur
un shard : 500/500). Sans lui, les runs restent en `grid_source=synthetic` :
c'est légal mais les courbes ne sont pas comparables aux runs mesurés.
Commande (streaming borné, ~1 Go de RAM) :
`python scripts/build_grid_sidecar.py --shards <les 17 shards du corpus> --out grid_sidecar.json --verify-index <key_index_cache>.json`
Le sidecar doit ensuite être poussé au même endroit que les manifests.

Toujours pas de précipitation, et tout lancement passe désormais par le
protocole notebook-natif (accord Molab post-unban : training attaché au
notebook, observabilité inline, pas de processus détaché/watchdog/tunnel).
