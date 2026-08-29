"""Shim — moved to vision_adapter/models/precompute_colab.py. Import path preserved during transition."""

def _shim():  # keep import-time side-effects identical
    from vision_adapter.models.precompute_colab import *  # noqa: F401,F403
    return None

_shim()

if __name__ == "__main__":
    from vision_adapter.models.precompute_colab import main as _main
    raise SystemExit(_main() if callable(_main) else 0)
