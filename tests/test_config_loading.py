import pytest
import yaml
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_perturbations.utils.load_yaml_config import load_yaml_config

def test_load_yaml_config_coerces_numeric_keys(tmp_path: Path) -> None:
    cfg = {
        "lr": "3e-4",
        "epochs": "4",
        "random_seed": "3",
        "name": "baseline",
        "already_int": 7,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = load_yaml_config(
        p,
        numeric_keys=["epochs", "random_seed", "already_int"],
        float_only=["lr"],
    )

    assert out["name"] == "baseline"
    assert out["epochs"] == 4 and isinstance(out["epochs"], int)
    assert out["random_seed"] == 3 and isinstance(out["random_seed"], int)
    assert out["lr"] == pytest.approx(3e-4) and isinstance(out["lr"], float)
    assert out["already_int"] == 7 and isinstance(out["already_int"], int)