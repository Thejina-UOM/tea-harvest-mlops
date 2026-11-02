"""Project paths and parameter loading.

I keep every tunable value in params.yaml so that DVC can detect when a parameter
changes and re-run only the affected pipeline stages.
"""

from functools import lru_cache
from pathlib import Path

import yaml

# Repository root = the folder that contains this "src" package.
ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"


@lru_cache(maxsize=1)
def load_params() -> dict:
    """Read params.yaml once and cache it. Raises a clear error if the file is missing."""
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"params.yaml not found at {PARAMS_PATH}")
    with PARAMS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
