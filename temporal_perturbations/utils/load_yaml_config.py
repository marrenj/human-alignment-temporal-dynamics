from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import yaml

from temporal_perturbations.paths import resolve_config_paths


def load_yaml_config(
    path: Path,
    numeric_keys: Iterable[str] | None = None,
    float_only: Iterable[str] | None = None,
) -> Mapping[str, object]:
    """
    Load a YAML config, resolve ``"paths:<key>"`` values, and coerce any keys
    listed in ``numeric_keys`` (or ``float_only``) into numeric types.
    Integer-looking floats (e.g. "4" or "3e0") are cast down to ints for the
    keys in ``numeric_keys``.
    """
    with Path(path).open("r") as handle:
        config = resolve_config_paths(yaml.safe_load(handle))

    numeric_keys = set(numeric_keys or [])
    float_only = set(float_only or [])

    for key in numeric_keys.union(float_only):
        value = config.get(key)
        if not isinstance(value, str):
            continue
        try:
            num = float(value)
        except (ValueError, TypeError):
            continue

        if key in numeric_keys and num.is_integer():
            config[key] = int(num)
        else:
            config[key] = num

    return config