"""Behavioral- and brain-alignment readouts of the from-scratch runs, shared by the figures.

Onsets early / mid / late are epochs 5 / 30 / 70 for the three ImageNet
classifiers and 5 / 13 / 28 for CLIP ViT-B/32. ResNet-50 uses BatchNorm-recalibrated evaluations. Brain
alignment = THINGS-fMRI (27 ROIs), corrected-preprocessing FR-RSA files.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_runs import CLIP_PREFIX, frrsa_dir, rdm1854_corrected_suffix
import make_fig_cross_architecture as CA

BN_RECAL_ARCHES = {'resnet50_imagenet'}

BRAIN_TAG = {
    'resnet50_imagenet': 'in1k_bilinear_bnrecal',
    'vitb16_imagenet': 'in1k_bicubic',
    'convnext_b_imagenet': 'in1k_bicubic',
    CLIP_PREFIX: 'clip',
}


def behavioral_curve(key, run):
    """{epoch: Spearman rho} vs the 1,854-object SPoSE RDM."""
    suffix = rdm1854_corrected_suffix(key)
    if key in BN_RECAL_ARCHES:
        suffix = '_bn_recal' + suffix
    tr = CA.load_traj(run, suffix)
    if tr is None:
        return {}
    return dict(zip(tr[0].astype(int).tolist(), tr[1].tolist()))


def brain_score(key, run, epoch):
    """Mean in-sample Spearman score over the 27 THINGS-fMRI ROIs."""
    p = (frrsa_dir(run)
         / f'inference_results_things_frrsa_{BRAIN_TAG[key]}_epoch{epoch}.json')
    if not p.is_file():
        return None
    with p.open() as f:
        blocks = json.load(f)
    scores = [b['insample_score'] for b in blocks.values()]
    assert len(scores) == 27 and None not in scores, f'{p}: incomplete ROI set'
    return float(np.mean(scores))
