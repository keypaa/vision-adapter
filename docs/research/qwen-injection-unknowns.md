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

## U4. Cause des 0 token en génération
`inputs_embeds` → 0 token (avec/sans cache, greedy/sampling), `input_ids`
texte-only OK, embeddings sains. Suspects restants : frontière
transformers×Qwen3.5-MTP, KV-cache GatedDeltaNet en fallback torch.
La sanity native (`apply_chat_template`, NEXT) tranche.
