# Qwen3.5 injection — UNKNOWNS

Par ordre d’importance. Chacun a son expérience de clôture dans
`qwen-injection-next.md`.

## U1. Règle d’expansion exacte des placeholders
Combien de `image_token_id` par image, et comment `image_token_lengths` /
`grid_thw` les dimensionnent (split `grid_thw.prod(-1) // merge²` vu dans le
code, à confirmer dans le processor). Bloque toute réécriture de `embeds_for`.

## U2. Le -33 % held-out mesure-t-il de la vision ?
Stratification manquante (par `g`, par `n_vis`, par longueur de réponse) +
aucun contrôle par permutation (`vis` mélangés, texte fixe). Sans ça,
shortcut de style/longueur non exclu.

## U3. mRoPE faux : invalide ou atténue ?
Le probe a appris *malgré* RoPE texte sur le span visuel. On ignore si le
protocole natif change l’ordre de grandeur du gain ou seulement sa marge.

## U4. Cause des 0 token en génération — RÉSOLU (2026-09-21, Molab PRO 6000)
Sorties projecteur hors-échelle : `vis rms=16.42, absmax=172` vs table
`rms=0.0131, absmax=0.082` (facteur ~1250×, pas de NaN/Inf). L’attention
sature → génération morte dans tous les régimes ; le loop `generate` est
sain (contrôle embeddings-table : 28 tokens). Le held-out -33 % a donc été
appris en régime saturé — une tête à l’échelle rallumerait la génération
et probablement une meilleure convergence.
Cartographie : `data/gen_norms.log` (ckpt final), helper de mesure dans
`scripts/colab_unsloth_test.py --debug-ids` (bloc `vis_embed`).
Suite : tête normalisée (LayerNorm × RMS table, variante A) + diag court,
voir `qwen-injection-next.md` NEXT-5.
Mise à jour (Molab, ckpt final) : normes seules n’ont pas rallumé la
génération. Table 2×2 : `input_ids` court/long OK (32 tok) ;
`inputs_embeds` court OK (28 tok), long (~600-740, texte OU visuel) → 0 token.
Verdict : le contenu visuel est exonéré, c’est la longueur via
`inputs_embeds` qui casse `generate` (frontière transformers×Qwen3.5,
cache/positions au prefill long). Training non affecté (forward sans cache).
Éval capacité viable = forced-decode/NLL ou serving stack, pas `generate`
transformers sur longs préfixes.
Percée (Molab, ckpt scalé-300, préfixe natif) : boucle manuelle KV-cache
maison (`manual_generate`, `--decode manual`) → 48 tokens générés là où
`generate()` rendait 0. Le bloqueur était bien `generate()`, pas le modèle.
Contenu généré : écho de prompt + hallucination (scène de bureau pour un
logo), pas de description — signal visuel faible ou ignoré à 300 steps
hors-domaine. Prochain test qui compte : image in-domain + comparaison
au texte assistant de sa propre row.
