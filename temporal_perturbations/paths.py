"""Filesystem locations of every external input and output, read from `pipeline/paths.yaml`.

Code never hard-codes an absolute path: it asks `path("<key>")`. Each key can be
overridden by an environment variable `TP_<KEY>` (upper-cased), which is how the
notebook and tests point scripts elsewhere without editing the YAML.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE = REPO_ROOT / "pipeline"
PATHS_FILE = PIPELINE / "paths.yaml"
DATA_DIR = PIPELINE / "data"

# Keys that default inside the repo, so a fresh clone can render figures from the
# shipped results bundle with no configuration at all.
_REPO_DEFAULTS = {
    "results_root": PIPELINE / "results",
    "figures_root": PIPELINE / "figures",
    "feature_cache": PIPELINE / "cache",
}


@lru_cache(maxsize=1)
def _configured():
    if not PATHS_FILE.exists():
        return {}
    with open(PATHS_FILE) as f:
        return yaml.safe_load(f) or {}


def path(key):
    """Resolve `key` from `TP_<KEY>`, then `pipeline/paths.yaml`, then the in-repo default."""
    env = os.environ.get(f"TP_{key.upper()}")
    if env:
        return Path(env).expanduser()
    value = _configured().get(key)
    if value:
        p = Path(value).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p
    if key in _REPO_DEFAULTS:
        return _REPO_DEFAULTS[key]
    raise KeyError(
        f"Path '{key}' is not configured. Set it in {PATHS_FILE} "
        f"(copy pipeline/paths.example.yaml) or export TP_{key.upper()}."
    )


def results_root():
    return path("results_root")


def resolve_config_paths(config):
    """Replace top-level `"paths:<key>"` string values in a loaded YAML config with `path(<key>)`.

    Lets the checked-in eval configs name an external dataset by its paths.yaml key
    instead of an absolute location. Keys are only resolved when present, so a
    config that is fully overridden on the command line needs no paths.yaml entry.
    """
    for k, v in config.items():
        if isinstance(v, str) and v.startswith("paths:"):
            config[k] = str(path(v[len("paths:"):]))
    return config
