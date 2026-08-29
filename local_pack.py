"""Shim — moved to vision_adapter/data/pack.py. Import path preserved during transition."""
from vision_adapter.data.pack import *  # noqa: F401,F403

if __name__ == "__main__":
    from vision_adapter.data.pack import main as _main
    raise SystemExit(_main())
