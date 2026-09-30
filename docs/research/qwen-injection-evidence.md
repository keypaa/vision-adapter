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

## Audit manifest live (2026-09-30, lecture-seule)

Méthode : `hf_hub_download` `keypa/vision-adapter-manifests`
(`train_manifest.jsonl` 82,3 Mo, `train_manifest_val.jsonl` 1,7 Mo, pas de
header), tokenisation Qwen3.5-2B sur sample 2 000/g (seed 0). Script :
`/tmp/opencode/manifest_audit.py` (hors repo, ré-exécutable). Aucun training.

Comptes : train 117 600 rows (agentic 52 924 / doc 52 908 / conv 11 768),
val 2 400 (agentic 1 076 / doc 1 092 / conv 232).

Markers (item 1 review externe) : 52 924/117 600 rows (45,0 %, toutes agentic)
contiennent des markers ; chaque row agentic = exactement 130 strings
(1 begin + 128 `|image|` + 1 end) = **265 tokens** (mesuré :
`|image|` ≈ 2 tokens `[8224, 1742]`). Chaîne causale :
`dataset.py:218-224` (contenu Sero verbatim) → `core.py:214` (tokenize sans
strip) → `core.py:285` (splice `[1:1+n_vis]` en plus). Compte fixe,
indépendant du vrai `n_vis` (16–16 653).
FIX 2026-09-30 : `strip_text_image_markers` dans `vision_adapter/core.py`
(retire le bloc avant tokenisation, texte sans markers inchangé), appliqué
dans `make_collate`. Pinné par `tests/test_marker_strip.py`. Toute run card
post-fix doit porter `extra.marker_strip=true` (comparaison aux courbes
historiques interdite sans cette mention, cf. R5).

Longueurs tokens user (p50/p90/p99/max) : agentic 283/410/455/494 ;
doc 34/46/62/541 ; conv 38/46/59/87. Réponses : agentic 12/12/25/38 ;
doc 5/9/20/706 ; conv 3/4/28/57.

Truncation simulée (`budget_text = 4096 - n_vis - 2`, sample 2 000/g) :
`n_vis` 400 et 2 000 → 0 coupe tous g ; `n_vis` 5 000 et 16 653 → budget
négatif, answer à 1 token, user perdu, `L` pire ~5,1k–17,9k > 4096
(`core.py:218-223`, `L` non cappé — overflow avéré, pas seulement théorique).
FIX 2026-09-30 : garde overflow dans `make_collate` (span visuel tronqué en
queue, `L <= max_len` garanti pour `max_len >= 4`, rows normales
byte-identiques ; `batch["n_vis"]` suit le span gardé, donc `embeds_for`,
mRoPE et natif restent cohérents). Pinné par
`tests/test_collate_overflow.py` (RED vu : `L=104 > 64` avant fix).

Dups image (item 5) : 8 653 clés `emb` dupliquées → 45 393 rows répétées
(38,6 % du train) ; mix : `doc` seul 6 060 clés / 40 128 rows (~12,8k images
uniques pour 52 908 rows), `conv` seul 2 593 clés / 5 265 rows, agentic 0,
cross-g 0. Top clé : 54 répétitions (`embeddings/504bea23d9ec40c41a2d.pt`).

Val (item 6) : 1 128/2 400 rows val (47,0 %) ont leur `emb` dans train ;
markers en val 1 076/2 400. `modal_train.py:70,442-453` charge ce val
(`val_loss` optimiste) ; `vision_adapter/train.py` streaming ne le charge
jamais ; disjonction image/épisode du val shard-level non vérifiée.

Bucketing (item avéré, code seul) : `stream.py:511` (`bucket_by_n_vis=True`
par défaut) vs `grok_probe_qwen.py:448` (plan propre non bucketé) vs
`modal_train.py:442` (`DataLoader shuffle=True`). Seul le path streaming
prod est bucketé.

Val disjoint — REBUILD 2026-09-30 (`scripts/rebuild_val_disjoint.py`,
`vision_adapter/manifest.py:disjoint_val_rows`, pinné par
`tests/test_disjoint_val.py`) : val 2 400 → **1 272 rows** (1 128 droppées,
overlap 0 vérifié par assert, header-first avec tags de provenance).
Biais de composition honnête : l'overlap était quasi-exclu doc/conv
(multi-questions par image) — le val disjoint restant est agentic 1 076 /
conv 70 / doc 126 (84,6 % agentic, non représentatif du mix 45/45/10).
1 seule clé dupliquée interne au nouveau val. Fichier prêt :
`/tmp/opencode/train_manifest_val_disjoint.jsonl` — push HF = étape humaine
(§ NEXT-6), upload sous **nouveau nom** (l'ancien fichier est conservé ;
`modal_train.py:VAL_MANIFEST_REL` pointe toujours l'ancien jusqu'au switch
explicite).

Grid synthétique — **MESURÉ, puis FIXÉ 2026-09-30**. Les deux endroits
(`core.py` `train_position_ids`, `native.py`) factorisaient `n_vis` en grille
carrée paire. Mesures (PC local, `/tmp/opencode/grid_distortion.py`,
échantillon 200 rows, RAM bornée 1,15 Go) :

- Vraie grille dérivable de (w,h) via le contrat preprocess : **400/400 rows
  exactes** contre le `n_vis` du key-index (`grid_recovery.py`).
- `grid_for_nvis` = vraie grille : **0/200**. Erreur d'aspect médiane **×3,0**
  (p90 ×4,4), **200/200 rows > 50 % d'erreur**.
- Pire : c'est une **inversion d'orientation**, pas une imprécision. Image
  364×784 px → vraie grille `[1,56,26]` (portrait), mRoPE recevait
  `[1,28,52]` (paysage). Cas UI/OCR/charts exactement.

Contre-exemple qui empêche un backfill par `n_vis` : `n_vis=364` couvre
**deux** grilles réelles, `(52,28)` et `(56,26)` (514 rows mesurées) —
`grid_uniqueness.py`. Il faut les dims par image, pas le compte de tokens.

FIX : `grid_from_dims(w, h)` (`core.py`) reconstruit la grille exacte sans
relancer le ViT ; `row_grids_for_batch` donne la priorité au `grid_thw`
mesuré portée par le batch, sinon synthétique ; mismatch grid/`n_vis` =
`ValueError` (bug de données, jamais masqué) ; `vision_adapter/grid_sidecar.py`
+ `scripts/build_grid_sidecar.py` construisent le sidecar emb→grille en
streaming borné (500/500 vérifiés contre le key-index). `grid_source` est
désormais dans la run card : une courbe `synthetic` n'est pas comparable à
une courbe `measured`. Pins : `tests/test_true_geometry.py`,
`tests/test_grid_sidecar.py` (182 tests verts).

`n_vis` par `g` (2026-09-30, PC local, 44 s, key-index 139k clés/13 Mo,
join 117 600/117 600, script `/tmp/opencode/nvis_per_g.py`,
plot `/tmp/opencode/nvis_per_g.png`) :

| `g` | rows | p50 | p90 | p99 | max | somme `n_vis` | part compute |
|---|---|---|---|---|---|---|---|
| agentic | 52 924 (45,0 %) | 364 | 364 | 378 | 380 | 17,4 M | 21,8 % |
| doc | 52 908 (45,0 %) | 960 | 1 175 | 5 184 | 16 598 | 57,8 M | 72,3 % |
| conv | 11 768 (10,0 %) | 368 | 814 | 1 036 | 1 610 | 4,7 M | 5,8 % |

Lecture : l'agentic est quasi-constant (~364, screenshots UI standardisés) —
jamais à risque de truncation (max 380). La long tail est portée par `doc`
(seule à dépasser 5 000). Le mélange effectif est ~22/72/6 en tokens visuels,
pas 45/45/10. Les 128 markers fixes ne correspondent à aucun `n_vis` réel
(364 en agentic).
