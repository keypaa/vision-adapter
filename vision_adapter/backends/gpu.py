"""vision_adapter/backends/gpu.py — GPU requirement helpers.

Only stages that strictly require CUDA (train, precompute) should call
require_gpu(). Stages like dataset/pack run fine on CPU.

Detection order (cheapest first):
  1) torch.cuda.is_available() when torch is importable — authoritative.
  2) nvidia-smi probe via subprocess — fallback when torch is absent.
Both are best-effort; a missing driver is treated as "no GPU".
"""
from __future__ import annotations

import shutil
import subprocess


def _torch_has_cuda() -> bool | None:
    """Return torch.cuda.is_available() if torch importable, else None."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return None


def _nvidia_smi_present() -> bool:
    """Whether nvidia-smi is on PATH and exits 0."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return r.returncode == 0
    except Exception:
        return False


def has_gpu() -> bool:
    """Whether a CUDA GPU is usable on this machine.

    True if torch reports CUDA or nvidia-smi succeeds. False otherwise.
    We accept ANY CUDA-capable GPU (T4, L4, A100, H100, 4090, …) — the stage
    itself should be hardware-agnostic and not hard-code a single SKU.
    """
    t = _torch_has_cuda()
    if t is not None:
        if t:
            return True
        return False
    return _nvidia_smi_present()


def _gpu_label() -> str:
    """Human-readable GPU label for diagnostics (best-effort)."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name} (CUDA {props.major}.{props.minor}, {props.total_memory // (1<<30)} GiB)"
    except Exception:
        pass
    # Fallback: nvidia-smi short query
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "unknown GPU"


def require_gpu(stage: str) -> None:
    """Assert a CUDA GPU is present; otherwise print a friendly warning and exit 2.

    Call this at the top of stages that strictly require CUDA (train, precompute).
    Dataset/pack must NOT call this.
    We accept ANY GPU — T4, L4, A100, 4090, etc. — the stage should not hard-code a SKU.
    """
    if has_gpu():
        return
    label = _gpu_label()
    msg = (
        f"[vision-adapter] '{stage}' requires a CUDA GPU, but none was detected (probed: {label}).\n"
        f"  - torch.cuda.is_available() returned False\n"
        f"  - nvidia-smi {'found but failed' if shutil.which('nvidia-smi') else 'not found on PATH'}\n"
        f"  This stage runs CUDA kernels and cannot proceed on CPU.\n"
        f"  Any CUDA-capable GPU works (T4, L4, A100, 4090, …) — no SKU requirement.\n"
        f"  Stages that do NOT require a GPU: dataset, pack\n"
        f"  Fix: attach a GPU (Colab Runtime -> Change runtime type -> GPU, or local NVIDIA driver + CUDA)\n"
        f"       or run the stage on a GPU machine.\n"
    )
    raise SystemExit(f"{msg}\nAborting '{stage}' (exit 2)")
