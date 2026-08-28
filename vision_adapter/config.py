"""vision_adapter/config.py — single typed source of truth for training constants.

Replaces the three scattered copies at:
  grok_probe_qwen.py:104  (LR/WARMUP/GRAD_CLIP/MAX_SEQ_LEN/EPOCHS)
  modal_probe.py:65       (LR/WARMUP/GRAD_CLIP/MAX_SEQ_LEN/DEFAULT_BS)
  modal_train.py:49       (BATCH_SIZE/LR/MAX_SEQ_LEN/EPOCHS)

No YAML, no Hydra until ~15 fields need composition. One frozen dataclass,
validated in __post_init__, logged verbatim with git SHA + manifest hash in the
JSONL header (Karpathy / Chip Huyen / Marin provenance rule — every run must be
comparable from its log alone).

Usage:
    from vision_adapter.config import TrainConfig, default_config, probe_config, config_header

    cfg = default_config()          # production: batch 8, save 200
    cfg = probe_config()            # L4 probe:   batch 16, save 500
    cfg = TrainConfig(lr=3e-4)      # override any field; frozen so replace via dataclasses.replace

    header = config_header(cfg, manifest_path="emb_cache/train_manifest.jsonl")
    # header is a JSON-serialisable dict with git_sha, manifest_sha256, timestamp,
    # python version, and every cfg field verbatim — write it as the first JSONL line.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# helpers — git SHA + manifest hash (best-effort, never crash the trainer)
# ---------------------------------------------------------------------------

def get_git_sha(repo_root: str | Path | None = None) -> str:
    """Best-effort `git rev-parse HEAD`. Returns 'unknown' if not a git repo."""
    # 1) explicit env override (CI / Modal where .git is absent)
    env = os.environ.get("VISION_ADAPTER_GIT_SHA") or os.environ.get("GIT_SHA")
    if env:
        return env.strip()
    # 2) git CLI
    try:
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def manifest_sha256(path: str | Path | None) -> str | None:
    """SHA-256 of a manifest file on disk. None if absent / unreadable."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _manifest_row_count(path: str | Path | None) -> int | None:
    if not path or not Path(path).is_file():
        return None
    try:
        c = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    c += 1
        return c
    except Exception:
        return None


def file_sha256(path: str | Path) -> str | None:
    """SHA-256 of any file on disk (parquet shard, manifest, etc.). None on miss."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _new_run_id() -> str:
    """Short run identifier: UTC timestamp + 6 hex chars of randomness."""
    import uuid

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{ts}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """All training constants in one validated object.

    Defaults match modal_train.py (production, batch 8). Probe overrides via
    probe_config() or dataclasses.replace(cfg, batch_size=16, ...).
    """

    # --- model geometry ---
    vision_dim: int = 4096              # MoonViT 2×2 merge flatten

    # --- optimisation ---
    lr: float = 5e-4
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # --- data / sequence ---
    max_seq_len: int = 4096
    epochs: int = 2
    batch_size: int = 8

    # --- telemetry / grok window (Baseten 900*64 = 57.6k) ---
    samples_per_baseten_grok: int = 900 * 64
    log_every: int = 1                  # modal_train: 1
    val_every: int = 250
    save_every: int = 200               # production; probe overrides to 500
    chart_every: int = 50
    status_every: int = 20
    plateau_check_every: int = 50
    plateau_window: int = 300
    plateau_rel_tol: float = 0.02
    ema_beta: float = 0.98
    spike_factor: float = 2.0
    spike_window: int = 100
    spike_min_history: int = 20

    # --- hardware caps (informational; enforced where the trainer runs) ---
    gpu_mem_cap_gib: float = 70.0       # A100 gate
    sys_ram_cap_gib: float = 400.0

    def __post_init__(self) -> None:
        if self.lr <= 0 or self.lr > 1:
            raise ValueError(f"lr must be in (0,1], got {self.lr}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >=0, got {self.warmup_steps}")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be >0, got {self.grad_clip}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be >0, got {self.max_seq_len}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be >0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be >0, got {self.batch_size}")
        if self.vision_dim <= 0:
            raise ValueError(f"vision_dim must be >0, got {self.vision_dim}")
        if not 0 < self.ema_beta < 1:
            raise ValueError(f"ema_beta must be in (0,1), got {self.ema_beta}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **overrides: Any) -> "TrainConfig":
        """Frozen-safe copy with overrides (alias for dataclasses.replace)."""
        return dataclasses.replace(self, **overrides)


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

def default_config(**overrides: Any) -> TrainConfig:
    """Production defaults (modal_train.py: batch 8, save 200, log 1)."""
    return TrainConfig(**overrides)


def probe_config(**overrides: Any) -> TrainConfig:
    """L4 / Colab probe preset: batch 16, sparser logging.

    Mirrors modal_probe.py: DEFAULT_BS=16, SAVE_EVERY=500, STATUS_EVERY=20,
    CHART_EVERY=50. Any field can still be overridden via kwargs.
    """
    base: dict[str, Any] = dict(batch_size=16, save_every=500, log_every=20)
    base.update(overrides)
    return TrainConfig(**base)


def colab_probe_config(**overrides: Any) -> TrainConfig:
    """Colab free-tier probe: batch 8, save 500 (matches grok_probe_qwen.py)."""
    base: dict[str, Any] = dict(batch_size=8, save_every=500, log_every=20)
    base.update(overrides)
    return TrainConfig(**base)


# ---------------------------------------------------------------------------
# header — the verbatim provenance record written as JSONL line 0
# ---------------------------------------------------------------------------

def config_header(
    cfg: TrainConfig,
    *,
    manifest_path: str | Path | None = None,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance header dict for the JSONL log.

    Every field of cfg is emitted verbatim so a run is comparable from its log
    alone (no code re-read). Includes git SHA, manifest hash/rows, timestamp,
    python/platform, and any caller-supplied extra (e.g. seed, device).
    """
    sha = get_git_sha()
    mhash = manifest_sha256(manifest_path)
    mrows = _manifest_row_count(manifest_path)
    rid = run_id or _new_run_id()
    header: dict[str, Any] = {
        "type": "config_header",
        "run_id": rid,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": sha,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": mhash,
        "manifest_rows": mrows,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": cfg.to_dict(),
    }
    if extra:
        header.update(extra)
    return header


__all__ = [
    "TrainConfig",
    "default_config",
    "probe_config",
    "colab_probe_config",
    "get_git_sha",
    "manifest_sha256",
    "file_sha256",
    "config_header",
]
