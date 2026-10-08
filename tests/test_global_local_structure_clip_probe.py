"""`analyze_global_local_structure.clip_probe_top1` must refuse a missing or older-recipe
CLIP ImageNet probe instead of silently dropping the run's top-1 ratio from Fig 3C-D.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from temporal_perturbations.inference import linear_probe_eval as lpe

SCRIPT = (Path(__file__).resolve().parent.parent / "pipeline" / "scripts" / "analysis"
          / "analyze_global_local_structure.py")
_spec = importlib.util.spec_from_file_location("analyze_global_local_structure", SCRIPT)
glsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glsa)

RUN = "clip_vitb32_image_noise_early"


def _write(root, record, epoch=5):
    (root / RUN).mkdir()
    (root / RUN / f"ood_probe_eval_epoch{epoch}.json").write_text(json.dumps(record))


def test_current_recipe_returns_top1(tmp_path):
    _write(tmp_path, {"c_selection": lpe.PROBE_C_SELECTION,
                      "feature_scaling": lpe.PROBE_FEATURE_SCALING,
                      "imagenet_val": {"top1_accuracy": 0.1094}})
    assert glsa.clip_probe_top1(tmp_path, RUN, 5) == 0.1094


@pytest.mark.parametrize("record", [
    {"c_selection": "val_sweep", "imagenet_val": {"top1_accuracy": 0.02}},
    {"c_selection": lpe.PROBE_C_SELECTION, "imagenet_val": {"top1_accuracy": 0.02}},
])
def test_older_recipe_raises(tmp_path, record):
    _write(tmp_path, record)
    with pytest.raises(ValueError, match="older recipe"):
        glsa.clip_probe_top1(tmp_path, RUN, 5)


def test_missing_record_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no CLIP ImageNet probe"):
        glsa.clip_probe_top1(tmp_path, RUN, 5)
