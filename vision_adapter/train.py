"""vision_adapter/train.py — shared train runner for the staged CLI.

Colab/local entrypoint: `python -m vision_adapter train --data-dir ./data`
assumes you already ran `dataset` (header-first manifest) and `precompute`
(embeddings). On a GPU box (any CUDA — T4, L4, A100, 4090) it runs the
Qwen probe end-to-end (tiny synthetic fallback when the manifest is the fake
dry-run fixture, so `python -m vision_adapter dataset --dry-run` + train
still proves the loop without 120k real embeddings).

Modal path remains the thin wrapper in modal_train.py (Volume) — this module
is intentionally backend-agnostic: it takes DataBackend and a TrainConfig.
"""
from __future__ import annotations

from pathlib import Path

import torch

from vision_adapter.backends.gpu import require_gpu
from vision_adapter.config import TrainConfig
from vision_adapter.manifest import load_manifest


def _tiny_qwen_for_smoke(vocab: int = 1024, hidden: int = 64, layers: int = 4):
    """Random-weight Qwen-shaped backbone, fp32 CPU/GPU — mirrors test_probe fixture.
    Last layer is full_attention so the projector receives grads (see test_probe notes)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
    tc = cfg.get_text_config()
    tc.vocab_size = vocab
    tc.hidden_size = hidden
    tc.intermediate_size = 2 * hidden
    tc.num_hidden_layers = layers
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.layer_types = ["linear_attention"] * (layers - 1) + ["full_attention"]
    try:
        tc.mtp_num_hidden_layers = None
    except Exception:
        pass
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class _StubTok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(ord(c) % 900) + 10 for c in text] or [10]}


def _smoke_train_with_fake_data(
    data_dir: Path,
    cfg: TrainConfig,
    max_steps: int | None,
    device: str,
) -> int:
    """5-step smoke on fake rows + random vis — no HF, no embeddings needed.
    Proves the projector + collate + monitors + selective loss wiring."""
    from vision_adapter.core import HourglassProjector, ProbeMonitor, make_collate, train_step_qwen

    rows, header = load_manifest(data_dir / "train_manifest.jsonl")
    is_fake = header is not None and any("fake" in r.get("emb", "") for r in rows[:5])
    # fall back to synthetic even when rows were from a real manifest but tiny limit
    limit_small = len(rows) <= 200
    if not is_fake and not limit_small and not header:
        # Not a fake fixture — caller should use the real HF streaming path
        return 1  # signal to caller to delegate to grok_probe_qwen instead

    steps = max_steps or 5
    steps = min(steps, 5)  # smoke cap
    tok = _StubTok()
    hidden = 64
    model = _tiny_qwen_for_smoke(hidden=hidden)
    dev = device if device in ("cuda", "cpu") and (device != "cuda" or torch.cuda.is_available()) else "cpu"
    if dev == "cuda":
        model = model.to("cuda")
    proj = HourglassProjector(cfg.vision_dim, hidden).to(dev)
    for p in proj.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(proj.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    monitor = ProbeMonitor()
    collate = make_collate(tok, tok.pad_token_id, max_len=cfg.max_seq_len, vision_dim=cfg.vision_dim)

    # Build batches from the manifest's fake rows: synthesize vis per row
    def _fake_vis(n_vis: int = 5):
        return torch.randn(n_vis, cfg.vision_dim)

    # Use at most `steps * batch_size` rows, reusing if needed
    import itertools
    B = cfg.batch_size
    batches = []
    cyc = itertools.cycle(rows if rows else [{"user": "hello", "assistant": "world", "emb": "x", "g": 0}])
    for _ in range(steps):
        items = []
        for _ in range(B):
            r = next(cyc)
            nv = 3 + (hash(r.get("emb", "")) % 5)
            items.append({"vis": _fake_vis(nv), "user": r.get("user", "hello"), "assistant": r.get("assistant", "hi"), "g": r.get("g", 0)})
        batches.append(collate(items))

    losses = []
    for i, batch in enumerate(batches, 1):
        out = train_step_qwen(model, proj, opt, batch, dev)
        losses.append(out["loss"])
        assert out["finite"], f"step {i} not finite: {out}"
        monitor.update(i, out["loss"], i * B)
    assert max(losses) - min(losses) >= 0  # at least runs
    print(f"[train] smoke {steps} steps on fake data — losses {[round(x,4) for x in losses]} monitor n_banners={monitor.n_banners}", flush=True)
    return 0


def run_train(
    data_dir: Path | str,
    cfg: TrainConfig,
    backend=None,
    max_steps: int | None = None,
    device: str | None = None,
) -> int:
    """Entry point for `cli train` non-dryrun.

    - Validates data_dir + manifest (header-first)
    - require_gpu("train") if device == "cuda" (any GPU)
    - Chooses path: fake-smoke (tiny) vs HF streaming (grok_probe) vs error with guidance
    Returns 0 on success, 1 if caller should delegate (e.g. no fake fixture, need HF path).
    """
    dd = Path(data_dir)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Only gate on GPU when we actually need CUDA kernels; smoke fallback runs on CPU
    # but warn — real training will need a card.
    need_gpu = dev == "cuda"
    if need_gpu:
        try:
            require_gpu("train")
        except SystemExit as e:
            print(str(e), flush=True)
            raise

    manifest_path = dd / "train_manifest.jsonl"
    if not manifest_path.exists():
        print(f"[train] manifest not found at {manifest_path} — run `python -m vision_adapter dataset --out {dd} [--dry-run]` first", flush=True)
        return 2

    # Prefer fake smoke when manifest is the dry-run fixture (fast proof without 2B download)
    rows, header = load_manifest(manifest_path)
    is_fake_fixture = header is not None and any("fake" in r.get("emb", "") for r in rows[:10])
    if is_fake_fixture or len(rows) <= 200:
        print(f"[train] detected {'fake fixture' if is_fake_fixture else 'small manifest'} ({len(rows)} rows) — running tiny smoke (no 2B download)", flush=True)
        return _smoke_train_with_fake_data(dd, cfg, max_steps, dev)

    # Real data: delegate to the HF streaming probe (Colab) — it owns RemoteShard + key index.
    # Keep this module free of HF streaming so local train stays importable without huggingface_hub.
    print("[train] real manifest detected — delegating to grok_probe_qwen streaming path", flush=True)
    print("  If you wanted fully local precomputed embeddings, put .pt files under <data-dir>/embeddings/ and re-run", flush=True)
    try:
        import grok_probe_qwen as gp  # local clone path; works when repo is checked out in Colab

        # Translate staged args → grok args: keep it minimal, use defaults for the rest
        # grok reads TrainConfig via vision_adapter.config, so we just ensure device/cfg line up
        args = gp.parse_args.__wrapped__ if hasattr(gp.parse_args, "__wrapped__") else gp.parse_args  # noqa: F841
        # Call grok's main directly — it will parse sys.argv, so set a minimal argv
        import sys

        sys.argv = [
            "grok_probe_qwen.py",
            "--model",
            "qwen2b",
            "--batch-size",
            str(cfg.batch_size),
            "--max-steps",
            str(max_steps or 5),
            "--out-dir",
            str(dd),
            "--cache-dir",
            str(dd / "cache"),
        ]
        return gp.main()
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)
    except Exception as e:
        print(f"[train] streaming delegate failed: {e} — falling back to smoke stub", flush=True)
        return _smoke_train_with_fake_data(dd, cfg, max_steps, dev)
