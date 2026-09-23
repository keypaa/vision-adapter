# TODO — Vision Adapter (état de référence)

Posture : **freeze d’entraînement** (aucun run > 1h sans gate MODE E), outils d’abord, YAGNI.

## Side quests (hors critique, moments creux)

- [x] Dataset card `keypa/vision-adapter-embeddings` (poussée 2026-09-23, vérifiée via API : README + cardData live). Licence marquée `unknown` — au owner de la fixer.
Conventions : `VISION_ADAPTER_PROJECT_STEWARD.md` + `VISION_ADAPTER_SPECIALIST_AGENT_PROMPTS.md`.
Preuve exigée pour tout claim. Détails d’exécution : `docs/superpowers/plans/2026-09-17-probe-4000-roadmap.md`.

## Fait (ne pas refaire)

- [x] Probe Qwen3.5-2B 4000 steps / 64k samples, 0 OOM (`keypa/vision-adapter-probe-checkpoints` : step4000, final, logs).
- [x] Mémoire : gate `bl2` 10M + micro-batching + fix grad-accum (`bfab5a1`), pins P0 CPU.
- [x] Held-out : 1.090 → 0.729 (-33 %, 60 rows, shards jamais entraînés).
- [x] Merge `master` (`fab07e5`), 109 tests verts, ruff clean.

## Reste — dans l’ordre

### A. Registre d’expériences + backfill
- [ ] Registre run cards JSONL (contrat steward : run ID, révision git, config, données, ckpts, métriques, held-out, interprétation, next).
- [ ] Backfill 3 cartes : diag-200, probe-1000, probe-4000 (preuves déjà sur HF + chat).
- [ ] DoD : toute question « qu’a montré le run X ? » répondue depuis le registre, pas depuis le chat.

### B. Décision méthode d’injection (verrou hero)
- [ ] Recherche web docs créateurs (Qwen natif early-fusion ; DeepSeek V4 attente visuelle ; recette serving vLLM/SGLang).
- [ ] Micro-exps minutes : `embeds_for` vs `visual_inject` vs path natif — même protocole, même métrique.
- [ ] DoD : méthode choisie + justification écrite + perdants documentés.

### C. Observabilité + guardrails
- [ ] Dashboard lisible (plotly : loss/EMA, gnorm, lr, VRAM, par bucket) remplaçant `probe_curves.png`.
- [ ] Guardrail no-learn auto-stop (définition chiffrée : ex. EMA sans progrès relatif X sur Y steps → arrêt).
- [ ] Cellule bootstrap versionnée unique + pré-vol (token, HEAD, disque, VRAM).
- [ ] DoD : un run aveugle > 1h devient impossible techniquement, pas par discipline.

### D. Éval génération viable
- [ ] Chemin serving (vLLM/SGLang, recette carte Qwen) OU preuve que `inputs_embeds`→0 token est contournable.
- [ ] Contexte : `Gen: ''` ×2 ckpts dans tous les régimes transformers ; path `input_ids` OK ; embeddings sains.
- [ ] DoD : générations comparables base vs final, qualitatif archivé.

### E. Bench + FlexAttention (après A–C)
- [ ] Bench `hf_transfer` vs Range, chiffré et archivé au registre.
- [ ] Micro-bench flex (unit / intégration / soak, kill criteria NEXT_STEPS §3).
- [ ] DoD : verdict flex GO/NO-GO avec mesures, pas d’avis.

### F. Gate hero MODE E → DeepSeek V4 FP4/FP8
- [ ] Gate steward complet (State/Evidence/Unknowns/Risks/Next) AVANT tout run pluri-heures.
- [ ] DoD : run hero lancé avec protocole, registre et guardrails — jamais « pour voir ».
