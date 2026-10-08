"""`run_clip_imagenet_ood_probe.py` stamps its records with `PROBE_C_SELECTION`, so its C
selection must be the probe core's: cold-started candidates, the shared iteration cap and
validation split.
"""

import importlib.util
import inspect
import logging
from pathlib import Path

import numpy as np

from temporal_perturbations.inference import linear_probe_eval as lpe

SCRIPT = (Path(__file__).resolve().parent.parent / "pipeline" / "scripts" / "evaluation"
          / "run_clip_imagenet_ood_probe.py")
_spec = importlib.util.spec_from_file_location("run_clip_imagenet_ood_probe", SCRIPT)
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)


def _data(seed=0, n=300, d=8, k=3):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n // k)
    x = rng.normal(size=(n, d)) + 0.8 * np.eye(k, d)[y]
    return x, y


def test_selection_matches_the_probe_core():
    x, y = _data()
    log = logging.getLogger("test")
    c_drv, sweep_drv = drv._select_C(x, y, "cpu", log)
    c_core, sweep_core = lpe._select_C(lpe._fit_and_score_torch, x, y, log, device="cpu")
    assert c_drv == c_core
    assert sweep_drv["C_grid"] == sweep_core["C_grid"]
    np.testing.assert_allclose(sweep_drv["val_top1"], sweep_core["val_top1"])


def test_sweep_record_carries_the_shared_settings():
    _, sweep = drv._select_C(*_data(), "cpu", logging.getLogger("test"))
    assert sweep["sweep_max_iter"] == lpe.PROBE_SWEEP_MAX_ITER
    assert sweep["val_seed"] == lpe.PROBE_VAL_SEED
    assert sweep["val_fraction"] == lpe.PROBE_VAL_FRACTION


def test_fit_has_no_warm_start():
    assert not {"W_init", "b_init"} & set(inspect.signature(drv._fit_probe_torch).parameters)
