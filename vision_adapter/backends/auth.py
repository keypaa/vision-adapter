"""vision_adapter/backends/auth.py — HF token resolution (CLI > env > Colab).

Does NOT assume Colab: google.colab.userdata is best-effort only.
Priority:
  1) --hf-token CLI arg
  2) $HF_TOKEN
  3) $HUGGING_FACE_HUB_TOKEN
  4) google.colab.userdata.get("HF_TOKEN") if available
"""
from __future__ import annotations

import os


def get_hf_token(cli_token: str | None = None) -> str | None:
    if cli_token:
        return cli_token
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(key)
        if v:
            return v
    try:
        from google.colab import userdata  # type: ignore

        v = userdata.get("HF_TOKEN")  # type: ignore[attr-defined]
        if v:
            return str(v)
    except Exception:
        pass
    return None


def set_hf_token_env(token: str | None) -> None:
    """Export token to env so huggingface_hub / datasets pick it up."""
    if token:
        os.environ["HF_TOKEN"] = token
        # huggingface_hub also reads HUGGING_FACE_HUB_TOKEN in some paths
        if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
