"""Shim — moved to vision_adapter/data/extract_moonvit_v2.py. Import path preserved during transition."""
from vision_adapter.data.extract_moonvit_v2 import *  # noqa: F401,F403

if __name__ == "__main__":
    import importlib, vision_adapter.data.extract_moonvit_v2 as _m
    _main = getattr(_m, "main", None)
    raise SystemExit(_main() if callable(_main) else 0)
