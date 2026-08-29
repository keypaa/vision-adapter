# ARCHITECTURE — Vision-Adapter over DeepSeek-V4-Flash

## 1. What we are building

A **frozen-backbone multimodal adapter**: we give the purely-textual 304B-param
`deepseek-ai/DeepSeek-V4-Flash-0731` LLM visual perception by training a tiny
67M-parameter projector on top of Kimi-K3's 401M-param MoonViT-V2 encoder,
following Baseten's public "GLM 5.2 with vision" recipe
(2-layer MLP, ~50M params, 66k images, batch 64, LR 5e-4, ~55 % MMMU-Pro).

```
  image ──[preprocess.py: navit_resize + norm]──► [N,3,14,14] pixel patches
            │
            ▼
      MoonViT-V2 (frozen, BF16, 401.2M params)     ← moonshotai/Kimi-K3, vision_tower.*
            │   27 pre-norm RMSNorm blocks, 2D RoPE + learnable 64×64 pos-emb,
            │   packed varlen flash-attn, no CLS token
            ▼
      merged tokens  [n_merged, 4, 1024]           ← 2×2 patch-merge (sd2_tpool)
            │   flatten → [n_merged, 4096]
            ▼
      HourglassProjector (TRAINABLE, 67,137,536 params)
            LN(4096) → Linear 4096→8192 → GELU → Linear 8192→4096
            │   drops each token to 4096-dim = DeepSeek hidden width
            ▼
      spliced into DeepSeek-V4's input embeddings at positions [1 : 1+n_merged]
            │   via `inputs_embeds=`, everything frozen except the projector
            ▼
      DeepSeek-V4-Flash (frozen, FP8/FP4 MoE, hidden 4096, CPU-offloaded)
```

Grokking is the mechanism that makes this tractable: loss is expected to
plateau until step ~900 (at batch 64 on 66k images), then collapse, after which
the projector is aligned and downstream behaviour persists without touching the
text model.

## 2. Verified facts (primary sources fetched in Phase 0/1)

| Fact | Value | Source |
|---|---|---|
| DeepSeek hidden size | **4096** | `config.json.hidden_size` |
| DeepSeek total params | 284 B (disk) | HF `safetensors.total` |
| DeepSeek quantisation | FP8 e4m3 (blocks) + int8 experts | `config.quantization_config` |
| MoonViT-V2 params | **401.2 M** | K3 model card + our state-dict count |
| MoonViT hidden size | **1024** | `vision_config.vt_hidden_size` |
| MoonVit patch / merge | 14 px / 2×2 → 28 px stride, token count = ⌈h/28⌉·⌈w/28⌉ | `kimi_k3_vision_processing.py` |
| Kimi projector input dim | 1024·4 = **4096** (matches DeepSeek) | `mm_projector.proj.0.weight` shape |
| Kimi projector output dim | 7168 (K3 hidden) | `mm_projector.proj.2.weight` shape |
| Vision-tower quantisation | **BF16** (carved out of the MXFP4 ignore-list) | `quantization_config.ignore` |

The 4096↔4096 coincidence means our projector (4096→8192→4096) preserves both
the visual token width and the text hidden width — no widening/narrowing
hazard at either end.

## 3. Components

| File | What it is | Owns |
|---|---|---|
| `vision_adapter/data/extract_moonvit_v2.py` | ONE-OFF. Pulled shards `model-00095/96-of-000096` from Kimi-K3 and saved `vision_tower.*` + `mm_projector.*` to `keypa/MoonViT-V2-Standalone`. | weight extraction |
| `vision_adapter/models/moonvit.py` | Standalone forward of the frozen vision tower (bit-exact vs. modeling_kimi_k3). | MoonViT |
| `vision_adapter/models/preprocess.py` | navit_resize (≤65536 patches, ≤7168 px/side, pad-to-28) + 0.5/0.5 normalise → packed `(pixel_values, grid_thws)`. | image → patches |
| `vision_adapter/data/agentic.py` | Downloads / resizes / renames the 82,829 agentic images by **positional join** into wave-ui-25k / ShowUI-desktop / aguvis-l1. | agentic image corpus |
| `vision_adapter/data/cauldron.py` | Cauldron pull — 6 download / 12 save subsets, writes `cauldron_manifest.jsonl`. | cauldron corpus |
| `vision_adapter/data/dataset.py` | Orchestrator: agentic + cauldron → header-first `train_manifest.jsonl` (`ORDER BY image`, pinned revisions). | dataset assembly |
| `vision_adapter/data/pack.py` | Packs embeddings into `SHARD_ROWS=1360` shards (`compression=None`, per-shard `sha256`). | shard packing |
| `vision_adapter/models/precompute.py` | Shared MoonViT precompute (`--backend local|modal`, `--revision` pin, `--patch-cap`). | embedding precompute |
| `vision_adapter/config.py` | Single frozen `TrainConfig` + `config_header` provenance (git SHA, manifest hash). | config / provenance |
| `vision_adapter/core.py` | Shared `HourglassProjector`, `make_collate`, `train_step`, monitors. | training core |
| `vision_adapter/manifest.py` | Versioned header-first manifest I/O (`manifest_version`, `ORDER BY`). | manifest |
| `vision_adapter/registry.py` | `runs.jsonl` experiment registry (one row per run, run_id-correlated). | registry |
| `vision_adapter/backends/base.py` + `local.py` + `modal.py` | `DataBackend` Protocol + Local/Modal implementations. | I/O backends |
| `vision_adapter/cli.py` | Sole staged entrypoint `python -m vision_adapter {dataset,precompute,pack,train,probe}`. | CLI |
| `modal_train.py` | Shim → `vision_adapter` training plane (dry-run memory gate → AdamW SFT). | training shim |

Note: **never put the embedding precompute and the training
loop in the same memory budget.** The whole point of the split is that the
vision tower is exercised once (offline) and the LLM only ever sees 4096-dim
cached tensors.

## 4. Load / persist contracts

* **MoonViT weights** —
  `safetensors` file at `keypa/MoonViT-V2-Standalone/moonvit_v2.safetensors`
  holds keys with their canonical Kimi prefix (`vision_tower.*`);
  `load_moonvit_from_safetensors` remaps them onto the local `_vt.*` namespace.
  The bug-fixed loader is pushed to HF so both Modal and Colab fetch the same
  code (`hf_hub_download(repo_id, 'moonvit.py', repo_type='model')`).

* **Image-mask hashing** — embedding file names are
  `sha1(relative_image_path)[:20] + '.pt'` where *relative* means relative to
  the `images/` root (e.g. `agentic/waveui_000123.png`).  This makes the
  manifest, the Modal precompute and the Colab precompute produce *identical*
  file names regardless of which host prefix is mounted at run time
  (`/data/images/...` vs `/content/drive/MyDrive/vision_adapter/images/...`).

* **Token/padding ground truth** (DeepSeek-V4-Flash-0731 tokenizer): `bos=0`
  `<｜begin▁of▁sentence｜>`, `eos=1` `<｜end▁of▁sentence｜>`, effective pad `≡ eos (1)`
  (a dedicated `<｜pad▁｜>` exists at id 2 but is not the configured pad). There is
  **no image/vocabulary sentinel token**, so vision tokens are injected strictly by
  `inputs_embeds` — never through `input_ids`. Positions `[1 : 1+n_vis]` hold the
  projected image embeddings; they carry a placeholder `input_ids` value only to
  keep the causal mask length correct, and are masked out of the loss.

* **Loss masking** — labels are -100 everywhere except the assistant answer
  (plus EOS). The BOS token, the `n_vis` image-embedding placeholder positions
  and the user prompt are never involved in the loss. Verified by
  `test_preprocess.py` and a manual collator-invariants pass.

* **Projector serialisation** — `safetensors` files
  `checkpoints/projector_step{N}.safetensors` and
  `checkpoints/projector_final.safetensors`, flat `state_dict` of the
  four `HourglassProjector` sub-modules (`ln`, `up`, `dn`; `act` has no
  params).

## 5. Memory budget proof (A100-80GB + 200 GB RAM)

| Component | On-disk | During train | Loaded from |
|---|---|---|---|
| DeepSeek-V4 weights | 155 GiB (native FP8/int8) | ≤ 70 GiB GPU (device_map="auto", rest CPU) | local HF cache |
| MoonViT-V2 weights | 0.80 GiB | **0** (precomputed offline) | — |
| HourglassProjector | 0.13 GiB bf16 | 0.13 GiB params + 0.27 GiB AdamW fp32 states + grads | fresh |
| Frozen activations | — | ~2–4 GiB w/ gradient checkpointing | — |
| Cached embeddings | ~5 GiB total | ~0.3 GiB per batch-8 (1200 tokens avg) | `/data/embeddings/` |

**Total peak GPU**: ≈ gates at `torch.cuda.max_memory_allocated() <
70 GiB`.  Exceeding that number means `train_dryrun` asserts and **aborts**
before any state is written.  DeepSeek is an MoE: even though 155 GiB lives on
disk, only its ≈ 13 B active parameters per token are swung into the 70 GiB
window by accelerate's CPU offload; vision precompute removes the 401M ViT
from the hot loop entirely, which is what makes this whole project boundable
on a single card.

## 6. Out of scope (documented on purpose)

* **RL / chain-of-thought grounding.** Baseten needed a second stage to
  restore reasoning. We ship the SFT grokking checkpoint only; RL on top is a
  follow-up project.
* **Grokking statistics** about exactly when the loss collapses are empirical;
  the training script logs loss every step so you can eyeball the cliff.
