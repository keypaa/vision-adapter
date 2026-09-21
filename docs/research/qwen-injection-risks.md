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
