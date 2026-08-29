"""Shim — moved to vision_adapter/data/agentic.py. Import path preserved during transition."""
from vision_adapter.data.agentic import *  # noqa: F401,F403, F403

if __name__ == "__main__":
    from vision_adapter.data.agentic import main as _main
    raise SystemExit(_main())
