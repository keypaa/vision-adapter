# Qwen3.5 injection — RISKS

## R1. Réécrire `embeds_for` avant validation
Réécrire l’injection vers le protocole natif sans la sanity génération
d’abord, c’est parier un refactor sur une lecture de code. Ordre imposé :
sanity (NEXT-1) puis refactor.

## R2. Doc tierce prise pour parole créateur
Le repo Megatron-Qwen3_5 n’est pas Qwen. Chaque point retenu doit recouper
le code transformers ou la doc officielle (fait pour les 4 points du spec ;
reste U1 à recouper dans le processor).

## R3. Positions visuelles et hash-MoE (hero)
Côté DeepSeek, le routage hash lit `input_ids` : des ids arbitraires aux
positions visuelles fausseraient le routage des premières couches.
À garder en tête pour la cible V4-Flash-0731, hors scope Qwen.

## R4. Benchmarks avant apportionnement
Écrire du Flex pour Qwen avant de savoir quelle couche domine (NEXT_STEPS §3),
ou bench Qwen en espérant un transfert DeepSeek : gaspillage probable.
Bench d’abord, code ensuite.

## R5. Tous les runs historiques incluent les markers (audit 2026-09-30)
Les 4 000 steps (resume 400→4000, scalé+mRoPE, held-out +11,9 %) ont entraîné
AVEC les 265 tokens markers par row agentic. Stripper change le setup :
comparer un futur run sans markers aux courbes historiques serait comparer
deux distributions différentes. Toute run card post-strip doit le noter
explicitement (champ à prévoir dans `experiments/run_cards.jsonl`).

## R6. Gates val non probantes à 47 % d'overlap (audit 2026-09-30)
`train_manifest_val.jsonl` recouvre train à 47 % par `emb` : un gate
`val_loss` sur ce split mesure en partie de la mémorisation. Ne pas verdir
de gate val sans split disjoint par `emb` (rebuild) ou sans le documenter
comme optimiste. Le held-out shards (`eval_heldout.py`) n'est pas innocenté
non plus (U7).
