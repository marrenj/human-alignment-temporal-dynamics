"""Run-name conventions and result locations shared by the figure and analysis scripts.

Every from-scratch run is named `{prefix}_{perturb_type}_{timing}` and its unperturbed
counterpart `{prefix}_baseline`.

Seed replicate. Every run exists for seed 42 (plain names) and seed 43 (the same names
+ `_seed43`, perturbed runs forked from the seed-43 baseline). `TP_SEED_SUFFIX=_seed43`
points every figure at the seed-43 family; unset, figures read seed 42.
"""

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from temporal_perturbations import paths  # noqa: E402

CLIP_PREFIX = 'clip_vitb32'

RESULTS = paths.results_root()
FIGURES = paths.path('figures_root')

SEED_SUFFIX = os.environ.get('TP_SEED_SUFFIX', '')
# `TP_PARTIAL=1` draws a replicate seed's figures from the inputs computed so far: a figure
# skips what is missing and says so in the figure, instead of stopping at the first gap.
PARTIAL = bool(os.environ.get('TP_PARTIAL'))
if PARTIAL and not SEED_SUFFIX:
    raise SystemExit('TP_PARTIAL applies only to a replicate seed (set TP_SEED_SUFFIX)')
PENDING = []


def runs_root():
    """Checkpoint root of the from-scratch runs (only the scripts that read checkpoints need it)."""
    return paths.path('runs_root')


def pending(what):
    """Record an input that is not computed yet; outside partial mode, fail on it."""
    if not PARTIAL:
        raise FileNotFoundError(what)
    if what not in PENDING:
        PENDING.append(what)
        print(f'PENDING: {what}')


def seeded(run):
    return f'{run}{SEED_SUFFIX}'


def baseline_run(prefix):
    return seeded(f'{prefix}_baseline')


def widen_for_seed(lim, data_lo, data_hi, what):
    """`lim` as hard-coded -- the figures' limits were fitted to seed 42's data -- or, for
    another seed, widened by a 5% margin wherever its data fall outside, printing the change."""
    lo, hi = lim
    if not SEED_SUFFIX or (lo <= data_lo and data_hi <= hi):
        return lim
    pad = 0.05 * (max(hi, data_hi) - min(lo, data_lo))
    new = (data_lo - pad if data_lo < lo else lo, data_hi + pad if data_hi > hi else hi)
    print(f'{what}: limits ({lo:.4g}, {hi:.4g}) -> ({new[0]:.4g}, {new[1]:.4g}) to fit the data')
    return new


def imagenet_val_curves_path():
    return RESULTS / f'imagenet_val_curves{SEED_SUFFIX}.json'


def structure_analysis_path():
    return RESULTS / f'global_local_structure_analysis{SEED_SUFFIX}.json'


def frrsa_dir(run):
    """Per-run FR-RSA results, `inference_results_{dataset}_frrsa[_{tag}]_epoch{E}.json`."""
    return RESULTS / 'frrsa' / run


# Eval-preprocessing tag each family's checkpoints were trained under -- a stdlib copy of
# `temporal_perturbations/data/eval_transforms.py::ARCH_EVAL_PREPROCESSING` (pinned by
# tests/test_arch_runs.py). Behavioral-alignment summaries carry it as a trailing tag,
# `..._rsa_summary{suffix}.json`.
PREPROCESSING_TAG = {
    'resnet50_imagenet': 'in1k_bilinear',
    'vitb16_imagenet': 'in1k_bicubic',
    'convnext_b_imagenet': 'in1k_bicubic',
    CLIP_PREFIX: 'clip',
}


def rdm1854_corrected_suffix(prefix):
    """Filename suffix of a family's 1854-image behavioral-alignment summary.

    CLIP composes two tags: its sweep scores the `cls` representation (768-d
    pre-projection CLS token), so `_cls` precedes the recipe tag. The timm families
    have one representation and carry the recipe tag alone.
    """
    rep = '_cls' if prefix == CLIP_PREFIX else ''
    return f'{rep}_{PREPROCESSING_TAG[prefix]}'


def frrsa_corrected_tag(prefix):
    """Tag of a family's corrected FR-RSA files, `..._frrsa_{tag}_epoch{E}.json`.

    ResNet-50 is read BatchNorm-recalibrated, as is every ResNet-50 value in the paper.
    """
    bn = '_bnrecal' if prefix == 'resnet50_imagenet' else ''
    return f'{PREPROCESSING_TAG[prefix]}{bn}'


# The linear-probe recipe every probe value in a figure must come from: a copy of
# `linear_probe_eval.PROBE_C_SELECTION` / `PROBE_FEATURE_SCALING` (cold-started C sweep,
# z-scored features), pinned by tests/test_arch_runs.py. A copy, not an import, so that
# drawing a figure does not load torch.
PROBE_C_SELECTION = 'val_sweep_v2'
PROBE_FEATURE_SCALING = 'standardized'

# Transfer and ImageNet-subset probes under that recipe; ResNet-50's after BatchNorm re-estimation.
PROBE_ROOT = RESULTS / 'linear_probe_eval_scratch_std'
PROBE_ROOT_BN_RECAL = RESULTS / 'linear_probe_eval_scratch_bn_recal_std'
# CLIP ViT-B/32's linear probe on the full ImageNet training set (768-d class-token features).
CLIP_IMAGENET_PROBE_DIR = RESULTS / 'imagenet_ood_eval_scratch_clip_cls'


def probe_root(run_name):
    return PROBE_ROOT_BN_RECAL if run_name.startswith('resnet50_imagenet') else PROBE_ROOT


def require_current_probe(record, where):
    """`record` itself if it was fit under the current probe recipe; otherwise raise."""
    found = (record.get('c_selection'), record.get('feature_scaling'))
    if found != (PROBE_C_SELECTION, PROBE_FEATURE_SCALING):
        raise ValueError(f'{where}: probe fit under an older recipe '
                         f'(c_selection={found[0]!r}, feature_scaling={found[1]!r})')
    return record


def clip_imagenet_probe(run_name, epoch):
    """CLIP ViT-B/32's ImageNet probe record at `epoch`; FileNotFoundError if absent."""
    path = CLIP_IMAGENET_PROBE_DIR / run_name / f'ood_probe_eval_epoch{epoch}.json'
    with path.open() as f:
        return require_current_probe(json.load(f), path)


def clip_imagenet_probe_epochs(run_name):
    """Every epoch with a CLIP ImageNet probe record for `run_name`."""
    return sorted(int(p.stem.split('epoch')[1])
                  for p in (CLIP_IMAGENET_PROBE_DIR / run_name).glob('ood_probe_eval_epoch*.json'))
