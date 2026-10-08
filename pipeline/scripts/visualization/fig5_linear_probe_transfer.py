"""Fig 5 -- linear-probe readouts after a one-epoch perturbation, at print size.

Definitions are in the paper's figure caption.

Drawn at the paper's text width (5.5 in) with the shared `paper_style` grammar
(colour = onset timing, marker = perturbation type, filled = input-side,
open = target-side, circled letters = cases T, K, L).

Panels
------
A  Cases T (ViT-B/16, image noise at epoch 70) and K (ConvNeXt-B, label
   shuffle at epoch 70): retained fraction (perturbed / baseline) of four
   readouts -- ImageNet top-1 of the model's own classifier, the ImageNet
   1000-class and 16-category linear probes, and the mean linear-probe top-1
   over 21 transfer datasets (the same 21 for every point in A-C). Each case has a log-scaled axis of optimizer
   steps inside epoch 71 (the first epoch after the perturbed one) next to a
   linear axis of whole epochs. Seed 43 (TP_SEED_SUFFIX=_seed43) has no
   sub-epoch runs or ImageNet probes: one epoch axis from epoch 70, with the
   own-classifier and transfer-probe readouts only.
B  Final-epoch retained fraction, one point per perturbed run: mean
   transfer-probe top-1 vs test accuracy (ImageNet top-1; CLIP ViT-B/32 has
   no classifier, so its ImageNet linear probe stands in).
C  Gap in mean transfer-probe top-1 (perturbed - baseline, percentage points)
   vs representational distance to baseline (1 - Spearman rho) at onset,
   onset + 1 and final epoch, wherever both runs have transfer probes.

Runs: onsets early / mid / late (5 / 30 / 70 of 90; CLIP 5 / 13 / 28 of 32).
ResNet-50 probes come from the BatchNorm-recalibrated
evaluation for the run and its baseline.
D  Case L (CLIP ViT-B/32, image noise at epoch 5): per-dataset linear-probe
   gap at the final epoch over the 21 datasets of the transfer mean,
   fine-grained datasets grouped first.

Partial mode (TP_PARTIAL=1, seed 43 only): a B or C point whose inputs are not
computed yet is left out of the panel and its statistics, and a grey note under
the figure counts what is missing.

Writes Fig6_linear_probe_generalization_defs_in_caption.{png,pdf} to figures_root.
"""

import argparse
import collections
import functools
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as S
from arch_runs import (CLIP_PREFIX, FIGURES, PARTIAL, RESULTS, SEED_SUFFIX, baseline_run,
                       clip_imagenet_probe, imagenet_val_curves_path, pending, probe_root,
                       require_current_probe, seeded, widen_for_seed)

S.apply()

STEM = 'Fig6_linear_probe_generalization_defs_in_caption'
STEM_APPENDIX = 'FigS_transfer_retention_distance'

# Only seed 42 has the sub-epoch re-runs (`*_recovery_dynamics`) and the
# ImageNet 1000-class / 16-category probes, so seed 43's panel A has no step
# axis and keeps the two readouts that exist at epoch resolution.
SUB_EPOCH = not SEED_SUFFIX

IMAGENET_CURVES = imagenet_val_curves_path()
# ResNet-50 final-epoch top-1 after BatchNorm recalibration, all 50,000 val images.
BN_RECAL_VAL_FULL = RESULTS / 'bn_recalibrated_val_50k'
# Own-classifier top-1 inside the first epoch after onset: the sub-epoch
# checkpoints were scored on a fixed 10,000-image val subset (10 per class), so
# step 0 and the baseline come from the same subset, not the 50,000-image curve.
SUBEPOCH_DIR = RESULTS / 'subepoch_imagenet_val'
VAL_10K_DIR = RESULTS / 'onset_val_10k'

M2M_ROOTS = {
    'resnet50_imagenet': RESULTS / 'model_to_model_rsa_scratch_pp-in1k_bilinear_bnrecal',
    'vitb16_imagenet': RESULTS / 'model_to_model_rsa_scratch_pp-in1k_bicubic',
    'convnext_b_imagenet': RESULTS / 'model_to_model_rsa_scratch_pp-in1k_bicubic',
    CLIP_PREFIX: RESULTS / 'model_to_model_rsa_scratch_clip_cls_pp-clip',
}

# Every dataset of the probe benchmark (PAPER_PROBE_DATASETS) that both runs of
# every plotted pair were probed on. Left out: PCam and the ImageNet family, never
# probed on these checkpoints -- so every mean in A-C covers the same 21.
TRANSFER_DATASETS = [
    'vtab/cifar10', 'vtab/cifar100', 'vtab/caltech101', 'vtab/dtd',
    'vtab/svhn', 'vtab/flowers', 'vtab/food101',
    'vtab/clevr_count_all', 'vtab/clevr_closest_object_distance',
    'vtab/kitti_closest_vehicle_distance', 'vtab/eurosat', 'vtab/pets',
    'vtab/resisc45', 'vtab/cars',
    'fgvc_aircraft', 'voc2007', 'country211', 'gtsrb', 'mnist',
    'renderedsst2', 'stl10',
]
# CIFAR-10/100 are also probed on their own (as a test readout), so only the
# other 12 mark a checkpoint as having been probed on the transfer suite.
_SUITE_MARKERS = [d for d in TRANSFER_DATASETS
                  if d not in ('vtab/cifar10', 'vtab/cifar100')]

DATASET_NAMES = {
    'vtab/pets': 'Oxford Pets',
    'vtab/food101': 'Food-101',
    'vtab/cars': 'Stanford Cars',
    'fgvc_aircraft': 'FGVC Aircraft',
    'vtab/flowers': 'Flowers-102',
    'vtab/dtd': 'DTD (textures)',
    'vtab/clevr_count_all': 'CLEVR object count',
    'vtab/caltech101': 'Caltech-101',
    'stl10': 'STL-10',
    'gtsrb': 'GTSRB (traffic signs)',
    'vtab/resisc45': 'RESISC45 (aerial)',
    'vtab/cifar100': 'CIFAR-100',
    'country211': 'Country211 (geolocation)',
    'vtab/pcam': 'PCam (histology)',
    'voc2007': 'PASCAL VOC 2007',
    'vtab/cifar10': 'CIFAR-10',
    'vtab/svhn': 'SVHN (house numbers)',
    'renderedsst2': 'Rendered SST-2 (text)',
    'vtab/eurosat': 'EuroSAT (satellite)',
    'mnist': 'MNIST (digits)',
    'vtab/clevr_closest_object_distance': 'CLEVR object distance',
    'vtab/kitti_closest_vehicle_distance': 'KITTI vehicle distance',
}

# The standard fine-grained transfer benchmarks (Kornblith et al. 2019).
FINE_GRAINED = ['vtab/pets', 'vtab/food101', 'vtab/cars', 'fgvc_aircraft',
                'vtab/flowers']

STEPS_PER_EPOCH = 834

CASE_T = {
    'label': 'T',
    'run': seeded('vitb16_imagenet_image_noise_late'),
    'recovery_run': 'vitb16_imagenet_image_noise_late_recovery_dynamics',
    'baseline': baseline_run('vitb16_imagenet'),
    'arch_prefix': 'vitb16_imagenet',
    'pert': 'image_noise',
    'onset': 70,
    'total_epochs': 90,
    'final_epoch': 89,
}

CASE_K = {
    'label': 'K',
    'run': seeded('convnext_b_imagenet_label_shuffle_late'),
    'recovery_run': 'convnext_b_imagenet_label_shuffle_late_recovery_dynamics',
    'baseline': baseline_run('convnext_b_imagenet'),
    'arch_prefix': 'convnext_b_imagenet',
    'pert': 'label_shuffle',
    'onset': 70,
    'total_epochs': 90,
    'final_epoch': 89,
}

CASE_L = {
    'label': 'L',
    'run': seeded(f'{CLIP_PREFIX}_image_noise_early'),
    'baseline': baseline_run(CLIP_PREFIX),
    'arch_prefix': CLIP_PREFIX,
    'onset': 5,
    'total_epochs': 32,
    'final_epoch': 31,
}

PERTURB_TYPES = ['image_noise', 'label_shuffle', 'uniform_images']

PERTURBED_RUNS = [seeded(f'{arch}_{pt}_{timing}') for arch in S.ONSET_EPOCH
                  for pt in PERTURB_TYPES for timing in S.TIMINGS]


FINAL_EPOCHS = {
    'resnet50_imagenet': 89,
    'vitb16_imagenet': 89,
    'convnext_b_imagenet': 89,
    CLIP_PREFIX: 31,
}

MINUS = '−'


# Data loaders

@functools.lru_cache(maxsize=None)
def _read_json(path):
    with open(path) as f:
        return json.load(f)


def load_imagenet_accuracy(run_name):
    raw = _read_json(str(IMAGENET_CURVES))['runs'][run_name]['val_acc1']
    epochs = sorted(int(e) for e in raw)
    return np.array(epochs), np.array([raw[str(e)] for e in epochs])


def load_imagenet_accuracy_at_epoch(run_name, epoch):
    raw = _read_json(str(IMAGENET_CURVES))['runs'][run_name]['val_acc1']
    return raw.get(str(epoch))


def load_final_top1(run_name, epoch):
    """Own-classifier top-1 at the final epoch: ResNet-50 after BatchNorm
    recalibration on all 50,000 val images, like its probes; the others are
    the logged 50,000-image value."""
    if run_name.startswith('resnet50_imagenet'):
        path = BN_RECAL_VAL_FULL / f'{run_name}_epoch{epoch}.json'
        rec = _read_json(str(path))
        if rec['val_per_class'] != 50:
            raise ValueError(f'{path}: {rec["val_per_class"]} per class, expected 50')
        return rec['bn_recalibrated']['val_acc1']
    return load_imagenet_accuracy_at_epoch(run_name, epoch)


def load_clip_imagenet_probe(run_name, epoch):
    """CLIP ViT-B/32's ImageNet linear-probe top-1 at `epoch` (current probe recipe only)."""
    return clip_imagenet_probe(run_name, epoch)['imagenet_val']['top1_accuracy']


def load_val_10k(run_name, epoch):
    """Own-classifier top-1 on the 10,000-image val subset (10 per class)."""
    d = _read_json(str(VAL_10K_DIR / f'{run_name}_epoch{epoch}.json'))
    if d['val_per_class'] != 10:
        raise ValueError(f'{run_name} epoch {epoch}: val_per_class {d["val_per_class"]}')
    return d['bn_recalibrated']['val_acc1']


def own_head_steps(case):
    """(steps, retained fraction) for the own classifier inside the first
    epoch after onset, all on the 10,000-image subset: step 0 = end of the
    perturbed epoch / baseline there; later steps / baseline at the end of
    the first epoch (the rule every step-axis readout follows)."""
    run, rec, bl = case['run'], case['recovery_run'], case['baseline']
    onset, first = case['onset'], case['onset'] + 1
    sub = _read_json(str(SUBEPOCH_DIR / f'{rec}.json'))
    if sub['val_per_class'] != 10 or any(s['epoch'] != first for s in sub['steps']):
        raise ValueError(f'{rec}: not 10/class or not all inside epoch {first}')
    b_first = load_val_10k(bl, first)
    # `step` is the 0-based batch index a checkpoint was saved after.
    steps = [0] + [s['step'] + 1 for s in sub['steps']]
    if steps[-1] != STEPS_PER_EPOCH:
        raise ValueError(f'{rec}: last checkpoint after {steps[-1]} steps, '
                         f'expected {STEPS_PER_EPOCH} (end of epoch {first})')
    fracs = ([load_val_10k(run, onset) / load_val_10k(bl, onset)]
             + [s['val_acc1'] / b_first for s in sub['steps']])
    return np.array(steps), np.array(fracs)


def load_probe_scores_all_datasets(run_name, epoch):
    """{dataset: probe block} at `epoch`; every block must come from the current probe
    recipe. ResNet-50's probes (`probe_root`) follow BatchNorm re-estimation, because a
    perturbed epoch overwrites its running statistics -- for the run and its baseline alike."""
    run_dir = probe_root(run_name) / run_name
    for pat in [f'linear_probe_eval_epoch{epoch:03d}.json',
                f'linear_probe_eval_epoch{epoch}.json']:
        fp = run_dir / pat
        if fp.exists():
            results = _read_json(str(fp)).get('results', {})
            for ds, block in results.items():
                require_current_probe(block, f'{fp} {ds}')
            return results
    return {}


def load_probe_score(run_name, dataset, epoch):
    results = load_probe_scores_all_datasets(run_name, epoch)
    return results[dataset]['score'] if dataset in results else None


def transfer_suite_probed(run_name, epoch):
    results = load_probe_scores_all_datasets(run_name, epoch)
    return any(ds in results for ds in _SUITE_MARKERS)


def _suite_missing(run_name, epoch):
    results = load_probe_scores_all_datasets(run_name, epoch)
    return [ds for ds in TRANSFER_DATASETS if ds not in results]


def load_transfer_mean(run_name, epoch):
    missing = _suite_missing(run_name, epoch)
    if missing:
        raise ValueError(f'{run_name} epoch {epoch}: probed on the transfer '
                         f'suite but missing {missing}')
    results = load_probe_scores_all_datasets(run_name, epoch)
    return np.mean([results[ds]['score'] for ds in TRANSFER_DATASETS])


# (run, epoch, reason) for every candidate point with no transfer-probe pair.
ABSENT = []

# Set by --skip-partial: a partly probed suite (a probe job still writing) is
# left out and listed in ABSENT instead of raising.
SKIP_PARTIAL = False


def transfer_pair(run_name, baseline, epoch, record=True):
    """(perturbed, baseline) mean over TRANSFER_DATASETS, or None -- recorded
    in ABSENT -- when either checkpoint was never probed on the suite, or
    (with SKIP_PARTIAL) was only partly probed."""
    run_ok = transfer_suite_probed(run_name, epoch)
    bl_ok = transfer_suite_probed(baseline, epoch)
    if not (run_ok and bl_ok):
        if not record:
            return None
        why = ' and '.join(w for w, ok in (('run', run_ok), ('baseline', bl_ok))
                           if not ok)
        if PARTIAL:
            pending(f'{run_name} transfer probes epoch {epoch} ({why} not probed)')
        else:
            ABSENT.append((run_name, epoch, f'{why} not probed on the transfer datasets'))
        return None
    if SKIP_PARTIAL:
        n = len(TRANSFER_DATASETS)
        partial = [f'{who} {n - len(missing)}/{n}' for who, missing in
                   (('run', _suite_missing(run_name, epoch)),
                    ('baseline', _suite_missing(baseline, epoch))) if missing]
        if partial:
            if record:
                ABSENT.append((run_name, epoch, 'transfer suite partly probed: '
                               + ', '.join(partial)))
            return None
    return load_transfer_mean(run_name, epoch), load_transfer_mean(baseline, epoch)


def load_m2m_rsa_at_epoch(run_name, arch_prefix, epoch,
                          stimulus='things_fmri_720'):
    root = M2M_ROOTS[arch_prefix] / run_name
    if not root.is_dir():
        return None
    for pat in [f'model_to_model_rsa_{stimulus}_epoch{epoch}.json',
                f'model_to_model_rsa_{stimulus}_epoch{epoch:03d}.json']:
        fp = root / pat
        if fp.exists():
            return _read_json(str(fp))['rsa']['spearman']['rho']
    return None


def _probe_epochs(run_name):
    eps = []
    for fp in (probe_root(run_name) / run_name).glob('linear_probe_eval_epoch*.json'):
        m = re.search(r'epoch(\d+)', fp.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


# Run discovery

def discover_perturbed_runs():
    """Yield (run_name, arch_prefix, perturb_type, timing, onset_epoch) for
    every perturbed run; each must have probe and distance results on disk
    (in partial mode, each result not on disk yet is recorded where it is loaded)."""
    for arch_prefix, timings in S.ONSET_EPOCH.items():
        for pt in PERTURB_TYPES:
            for timing, onset in timings.items():
                run_name = seeded(f'{arch_prefix}_{pt}_{timing}')
                for d in (probe_root(run_name) / run_name,
                          M2M_ROOTS[arch_prefix] / run_name):
                    if not d.is_dir() and not PARTIAL:
                        raise FileNotFoundError(d)
                yield run_name, arch_prefix, pt, timing, onset


def onset_color(timing):
    return S.ONSET_COLORS[timing]


def _scatter_kw(pert, color):
    """S.scatter_kw with see-through open markers: a white-faced open circle
    drawn over a filled diamond reads as a (non-existent) open diamond."""
    kw = S.scatter_kw(pert, color, s=15)
    if not S.PERT_MARKER[pert][1]:
        kw['facecolors'] = 'none'
    return kw



# Layout helpers (positions in inches from the top-left corner)

def _add_axes(fig, left, top, width, height):
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def _fig_text(fig, x, top, s, **kw):
    W, H = fig.get_size_inches()
    return fig.text(x / W, 1 - top / H, s, **kw)


def _right_of(fig, artist, pad_pt=3.5):
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi + pad_pt / 72


def _header(fig, x, top, letter, title):
    """Bold panel letter + plain title on one baseline."""
    t = _fig_text(fig, x, top, letter, fontsize=S.FS_LETTER,
                  fontweight='bold', ha='left', va='baseline')
    x1 = _right_of(fig, t)
    _fig_text(fig, x1, top, title, fontsize=S.FS_TITLE, ha='left',
              va='baseline')
    return x1


def _fit_lim(what, lim, ticks, values):
    """`widen_for_seed`, with `ticks` extended at their spacing to the new limits."""
    if PARTIAL and len(values) == 0:
        return lim, ticks
    new = widen_for_seed(lim, min(values), max(values), what)
    if new is lim or ticks is None:
        return new, ticks
    step = ticks[1] - ticks[0]
    ticks = list(ticks)
    while ticks[0] - step >= new[0]:
        ticks.insert(0, round(ticks[0] - step, 6))
    while ticks[-1] + step <= new[1]:
        ticks.append(round(ticks[-1] + step, 6))
    return new, ticks


def _case_tag_fig(fig, x, top, letter):
    """Circled case letter centred at a figure position."""
    W, H = fig.get_size_inches()
    return fig.text(x / W, 1 - top / H, letter, ha='center', va='center',
                    fontsize=S.FS_NOTE + 0.5, fontweight='bold', zorder=12,
                    bbox=dict(boxstyle='circle,pad=0.18', facecolor='white',
                              edgecolor='#111111', linewidth=0.7))


# Panel A: four-readout trajectories for cases T and K

# Shades of the late-onset orange, dark -> light, one per readout: both
# cases are late onsets, so orange is the only hue the grammar allows here.
READOUT_STYLES = {
    'own_head':  {'color': '#5c230b', 'marker': 's', 'ms': 3.6,
                  'label': 'ImageNet top-1, own classifier'},
    'probe_1k':  {'color': '#a8421a', 'marker': '^', 'ms': 4.2,
                  'label': 'ImageNet 1000-class linear probe'},
    'probe_16':  {'color': '#eb6834', 'marker': 'v', 'ms': 4.2,
                  'label': 'ImageNet 16-category linear probe'},
    'mean_vtab': {'color': '#f4a278', 'edge': '#c4602c', 'marker': 'D', 'ms': 4.0,
                  'label': 'Transfer-probe mean (21 datasets)'},
}
if not SUB_EPOCH:
    READOUT_STYLES = {k: READOUT_STYLES[k] for k in ('own_head', 'mean_vtab')}

STEP0_X = 1.0   # step 0 sits where step 1 would on the log axis, behind a break


def _readout_trajectories(case):
    """{readout: {'step': (steps, fracs), 'epoch': (epochs, fracs)}}.

    'step' holds checkpoints inside the first epoch after the perturbed one,
    placed at the optimizer-step count they were saved after (step 0 = end of
    the perturbed epoch; the last, at STEPS_PER_EPOCH, ends epoch onset+1);
    'epoch' holds whole-epoch checkpoints from onset+1 on. The step axis stays
    on the recovery re-run past step 0: the original run's epoch-(onset+1)
    checkpoint sits at the same step count but comes from a separate training
    execution, so it is drawn on the epoch axis only. Retained fraction =
    perturbed / baseline at the same epoch; step-level checkpoints are divided
    by the baseline at the end of epoch onset+1, the nearest baseline
    checkpoint.
    """
    run, rec, bl = case['run'], case['recovery_run'], case['baseline']
    onset, final = case['onset'], case['final_epoch']
    first = onset + 1

    full_epochs = set(_probe_epochs(run))
    oh_epochs, _ = load_imagenet_accuracy(run)
    full_epochs |= {int(e) for e in oh_epochs if int(e) >= onset}
    full_epochs = sorted(full_epochs & set(range(onset, final + 1)))
    sub_labels = sorted(_probe_epochs(rec))

    def frac(p, b):
        return p / b if (p is not None and b is not None and b != 0) else None

    out = {}
    for readout in READOUT_STYLES:
        steps, sfr, eps, efr = [], [], [], []
        for label in full_epochs + sub_labels:
            is_sub = label >= 1000
            if readout == 'own_head':
                if is_sub:
                    continue
                f = frac(load_imagenet_accuracy_at_epoch(run, label),
                         load_imagenet_accuracy_at_epoch(bl, label))
            elif readout in ('probe_1k', 'probe_16'):
                ds = 'imagenet1k' if readout == 'probe_1k' else 'imagenet1k_16superclass'
                f = frac(load_probe_score(rec if is_sub else run, ds, label),
                         load_probe_score(bl, ds, first if is_sub else label))
            else:
                if is_sub:
                    continue
                pair = transfer_pair(run, bl, label, record=False)
                f = frac(*pair) if pair else None
            if f is None:
                continue
            if readout == 'own_head':
                if label >= first:
                    eps.append(label)
                    efr.append(f)
                continue
            if is_sub:
                # Label = epoch * 1000 + the 0-based batch index.
                steps.append(label - first * 1000 + 1)
                sfr.append(f)
                continue
            if label == onset:
                steps.append(0)
                sfr.append(f)
            if label >= first:
                eps.append(label)
                efr.append(f)
        if readout == 'own_head':
            steps, sfr = own_head_steps(case)
        elif max(steps, default=0) not in (0, STEPS_PER_EPOCH):
            raise ValueError(f'{rec} {readout}: last step-axis checkpoint after '
                             f'{max(steps)} steps, expected {STEPS_PER_EPOCH}')
        so = np.argsort(steps)
        out[readout] = {'step': (np.array(steps)[so], np.array(sfr)[so]),
                        'epoch': (np.array(eps), np.array(efr))}
    return out


def _epoch_trajectories(case):
    """Panel A without a step axis: {readout: {'epoch': (epochs, fracs)}}
    from the perturbed epoch on -- the own classifier at every logged epoch
    through the final one, the transfer-probe mean at onset, onset + 1 and
    the final epoch -- each perturbed / baseline at the same epoch, as in
    `_readout_trajectories`."""
    run, bl = case['run'], case['baseline']
    onset, final = case['onset'], case['final_epoch']
    own_eps = list(range(onset, final + 1))
    unlogged = [(r, e) for r in (run, bl) for e in own_eps
                if load_imagenet_accuracy_at_epoch(r, e) is None]
    if unlogged:
        raise ValueError(f'{IMAGENET_CURVES}: no val_acc1 for {unlogged}')
    own = [load_imagenet_accuracy_at_epoch(run, e) / load_imagenet_accuracy_at_epoch(bl, e)
           for e in own_eps]
    probe_eps = [onset, onset + 1, final]
    pairs = [transfer_pair(run, bl, e, record=False) for e in probe_eps]
    unprobed = [e for e, pair in zip(probe_eps, pairs) if pair is None]
    if unprobed:
        raise FileNotFoundError(f'{run} / {bl}: no transfer-probe pair at epochs '
                                f'{unprobed} under {probe_root(run)}')
    return {'own_head': {'epoch': (np.array(own_eps), np.array(own))},
            'mean_vtab': {'epoch': (np.array(probe_eps),
                                    np.array([p / b for p, b in pairs]))}}


def _step_axis_break(ax, xb=3.2):
    """Two short slashes on the step axis between step 0 and step 10."""
    tr = ax.get_xaxis_transform()
    for x in (xb, xb * 1.4):
        ax.plot([x / 1.13, x * 1.13], [-0.06, 0.06], color='#111111', lw=0.9,
                transform=tr, clip_on=False, zorder=6)


def _draw_case(ax_s, ax_e, case, trajs, ylim, yticks):
    """`ax_s` None: no step axis -- the epoch axis starts at the perturbed
    epoch and carries the y axis and the onset star."""
    target_side = case['pert'] in S.TARGET_SIDE
    ls = S.pert_linestyle(case['pert'])
    axes = [ax for ax in (ax_s, ax_e) if ax is not None]
    for ax in axes:
        ax.axhline(1.0, color=S.BASELINE_COLOR, lw=0.8, zorder=1)
    for readout, st in READOUT_STYLES.items():
        mk = dict(marker=st['marker'], markersize=st['ms'],
                  markeredgecolor=st.get('edge', st['color']), markeredgewidth=0.9,
                  markerfacecolor='white' if target_side else st['color'])
        if ax_s is not None:
            steps, sf = trajs[readout]['step']
            x = np.where(steps == 0, STEP0_X, steps).astype(float)
            if len(x):
                ax_s.plot(x, sf, color=st['color'], lw=S.LW_DATA, ls=ls, zorder=3,
                          **mk)
        eps, ef = trajs[readout]['epoch']
        if len(eps):
            if readout == 'own_head':
                # Consecutive epochs: markers on every one fuse into a bar,
                # so only the ends carry the marker; the line joins them all.
                mk = dict(mk, markevery=[0, len(eps) - 1])
            ax_e.plot(eps, ef, color=st['color'], lw=S.LW_DATA, ls=ls,
                      zorder=5 if readout == 'mean_vtab' else 3, **mk)

    if ax_s is not None:
        ax_s.set_xscale('log')
        ax_s.set_xlim(0.72, 1250)
        ax_s.xaxis.set_major_locator(FixedLocator([STEP0_X, 10, 100, STEPS_PER_EPOCH]))
        ax_s.xaxis.set_minor_locator(NullLocator())
        ax_s.set_xticklabels(['0', '10', '100', str(STEPS_PER_EPOCH)])
        ax_s.set_xlabel(f'Steps into epoch {case["onset"] + 1} (log scale)')
        _step_axis_break(ax_s)

    ax_e.set_xlim(case['onset'] - 0.5, case['final_epoch'] + 1.2)
    first_tick = case['onset'] + (1 if ax_s is not None else 0)
    ax_e.set_xticks([first_tick, 80, case['final_epoch']])
    ax_e.set_xlabel('Epoch')
    if ax_s is not None:
        ax_e.tick_params(axis='y', left=False, labelleft=False)
        ax_e.spines['left'].set_visible(False)
    for ax in axes:
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)

    star_ax, star_x = (ax_s, STEP0_X) if ax_s is not None else (ax_e, case['onset'])
    star_ax.plot([star_x], [1.0], transform=star_ax.get_xaxis_transform(),
                 marker='*', markersize=S.MS_STAR, color=S.ONSET_COLORS['late'],
                 markeredgewidth=0, clip_on=False, zorder=7)


def panel_a(pairs):
    ylims = [((0.86, 1.02), [0.88, 0.92, 0.96, 1.0]),
             ((-0.04, 1.08), [0.0, 0.25, 0.5, 0.75, 1.0])]
    end_of_first, transfer_final = [], []
    for axes, case, (ylim, yt) in zip(pairs, (CASE_T, CASE_K), ylims):
        ax_s, ax_e = axes if SUB_EPOCH else (None, *axes)
        trajs = _readout_trajectories(case) if SUB_EPOCH else _epoch_trajectories(case)
        ylim, yt = _fit_lim(f'panel A, case {case["label"]}, y', ylim, yt,
                            np.concatenate([f for t in trajs.values()
                                            for _, f in t.values()]))
        _draw_case(ax_s, ax_e, case, trajs, ylim, yt)
        for readout in [r for r in READOUT_STYLES if r != 'mean_vtab']:
            eps, ef = trajs[readout]['epoch']
            end_of_first.append(ef[eps == case['onset'] + 1][0])
        eps, ef = trajs['mean_vtab']['epoch']
        transfer_final.append(ef[eps == case['final_epoch']][0])
    pairs[0][0].set_ylabel('Retained fraction\n(perturbed / baseline)')
    return end_of_first, transfer_final


# Panel B: final-epoch probe retention vs test retention

def _load_or_none(load, run_name, epoch):
    """`load(run_name, epoch)`, or None in partial mode where it is not on disk yet."""
    try:
        return load(run_name, epoch)
    except FileNotFoundError:
        if not PARTIAL:
            raise
        return None


# Cached: in partial mode make_figure reads the points first, to size the notes.
@functools.cache
def _final_epoch_points():
    """(points, {input not computed yet: runs left out}), the second only in partial mode."""
    pts, left_out = [], collections.Counter()
    for run_name, arch_prefix, pt, timing, onset in discover_perturbed_runs():
        bl = baseline_run(arch_prefix)
        final = FINAL_EPOCHS[arch_prefix]
        if arch_prefix == CLIP_PREFIX:
            load, what = load_clip_imagenet_probe, 'CLIP ImageNet linear-probe top-1'
        else:
            load, what = load_final_top1, 'ImageNet top-1'
        test_p, test_b = (_load_or_none(load, r, final) for r in (run_name, bl))
        if PARTIAL and (test_p is None or test_b is None):
            missing = [r for r, v in ((run_name, test_p), (bl, test_b)) if v is None]
            pending(f'{" and ".join(missing)} {what} epoch {final}')
            left_out[what] += 1
            continue
        if test_p is None or test_b is None or test_b == 0:
            print(f'WARNING: panel B leaves out {run_name}: epoch-{final} test '
                  f'top-1 {test_p}, baseline {bl} {test_b} ({IMAGENET_CURVES})')
            continue
        pair = transfer_pair(run_name, bl, final)
        if pair is None:
            if PARTIAL:
                left_out['transfer probes'] += 1
            continue
        probe_p, probe_b = pair
        pts.append(dict(run=run_name, arch=arch_prefix, pert=pt, timing=timing,
                        onset=onset, test=test_p / test_b,
                        probe=probe_p / probe_b))
    return pts, left_out


def _case_l_point(pts, panel):
    for q in pts:
        if q['run'] == CASE_L['run']:
            return q
    if PARTIAL:
        return None
    absent = [(epoch, why) for run, epoch, why in ABSENT if run == CASE_L['run']]
    raise ValueError(f'panel {panel}: no point for case L ({CASE_L["run"]}); '
                     f'absent: {absent}')


def panel_b(ax):
    pts, _ = _final_epoch_points()
    xlim, xticks = _fit_lim('panel B, x', (0.2, 1.03), [0.25, 0.5, 0.75, 1.0],
                            [p['test'] for p in pts])
    ylim, yticks = _fit_lim('panel B, y', (0.6, 1.03), [0.6, 0.7, 0.8, 0.9, 1.0],
                            [p['probe'] for p in pts])
    lo, hi = ylim[0], max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color=S.REF_COLOR, lw=S.LW_REF,
            ls=(0, (1.5, 1.5)), zorder=1)
    ax.text(0.80, 0.70, 'y = x', ha='left', va='bottom',
            fontsize=S.FS_NOTE, color=S.REF_COLOR)
    ax.axhline(1.0, color=S.BASELINE_COLOR, lw=0.6, zorder=1)
    ax.axvline(1.0, color=S.BASELINE_COLOR, lw=0.6, zorder=1)
    for p in pts:
        ax.scatter(p['test'], p['probe'], zorder=3,
                   **_scatter_kw(p['pert'], onset_color(p['timing'])))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel('ImageNet top-1 (retained fraction);\n'
                  'CLIP ViT-B/32: linear probe')
    ax.set_ylabel('Transfer-probe mean\n(retained fraction)')

    p = _case_l_point(pts, 'B')
    if p is not None:
        S.case_tag(ax, p['test'] - 0.045, p['probe'] + 0.05, CASE_L['label'])
        ax.plot([p['test'] - 0.035, p['test'] - 0.004], [p['probe'] + 0.038, p['probe'] + 0.005],
                color='#111111', lw=0.5, zorder=11)

    clip = [p for p in pts if p['arch'] == CLIP_PREFIX]
    sup = [p for p in pts if p['arch'] != CLIP_PREFIX]
    # Two points always give r = +-1, so a partial render needs three.
    if PARTIAL and len(clip) < 3:
        r_clip = np.nan
        clip_text = f'CLIP,\n{len(clip)} run{"" if len(clip) == 1 else "s"}:\ntoo few for r'
    else:
        r_clip = stats.pearsonr([p['test'] for p in clip], [p['probe'] for p in clip])[0]
        clip_text = f'CLIP,\n{len(clip)} runs:\nr = {r_clip:.3f}'
    s_lo = min(min(p['test'] for p in sup), min(p['probe'] for p in sup))
    s_hi = max(max(p['test'] for p in sup), max(p['probe'] for p in sup))
    ax.text(0.04, 0.60, clip_text,
            transform=ax.transAxes, ha='left', va='center', fontsize=S.FS_NOTE,
            color=S.NOTE_COLOR, linespacing=1.2)
    return pts, r_clip, (s_lo, s_hi)


# Panel C: representational distance vs transfer-probe gap

@functools.cache
def _distance_gap_points():
    """(points, {input not computed yet: points left out}), the second only in partial mode."""
    pts, left_out = [], collections.Counter()
    for run_name, arch_prefix, pt, timing, onset in discover_perturbed_runs():
        bl = baseline_run(arch_prefix)
        for epoch in (onset, onset + 1, FINAL_EPOCHS[arch_prefix]):
            pair = transfer_pair(run_name, bl, epoch)
            if pair is None:
                if PARTIAL:
                    left_out['transfer probes'] += 1
                continue
            rho = load_m2m_rsa_at_epoch(run_name, arch_prefix, epoch)
            if rho is None and PARTIAL:
                pending(f'{run_name} representational distance epoch {epoch}')
                left_out['representational distance'] += 1
                continue
            if rho is None:
                ABSENT.append((run_name, epoch, 'no representational distance'))
                continue
            pts.append(dict(run=run_name, arch=arch_prefix, pert=pt,
                            timing=timing, onset=onset, epoch=epoch,
                            dist=1.0 - rho, gap=100 * (pair[0] - pair[1])))
    return pts, left_out


def panel_c(ax):
    pts, _ = _distance_gap_points()
    ax.axhline(0, color=S.BASELINE_COLOR, lw=0.6, zorder=1)
    for p in sorted(pts, key=lambda p: p['pert'] == 'label_shuffle'):
        ax.scatter(p['dist'], p['gap'], zorder=3, alpha=0.8,
                   **_scatter_kw(p['pert'], onset_color(p['timing'])))
    if PARTIAL and len(pts) < 3:
        rho = np.nan
    else:
        rho = stats.spearmanr([p['dist'] for p in pts], [p['gap'] for p in pts])[0]
    if not pts:
        ax.text(0.5, 0.5, 'not yet computed', transform=ax.transAxes, ha='center',
                va='center', fontsize=S.FS_NOTE, color=S.NOTE_COLOR)
    xlim, _ = _fit_lim('panel C, x', (-0.03, 1.0), None, [p['dist'] for p in pts])
    ylim, yticks = _fit_lim('panel C, y', (-70, 5), [-60, -40, -20, 0],
                            [p['gap'] for p in pts])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel('Representational distance\n'
                  f'to baseline (1 {MINUS} Spearman ρ)')
    ax.set_ylabel('Transfer-probe mean,\ngap to baseline (pp)')

    p = _case_l_point([q for q in pts if q['epoch'] == CASE_L['final_epoch']], 'C')
    if p is not None:
        S.case_tag(ax, p['dist'] + 0.07, p['gap'] - 9, CASE_L['label'])
        ax.plot([p['dist'], p['dist'] + 0.05], [p['gap'], p['gap'] - 6.5],
                color='#111111', lw=0.5, zorder=11)
    return pts, rho


# Panel D: case L per-dataset gap at the final epoch

def panel_d(ax, head_x):
    """`head_x`: axes-fraction x where the group headers start (left-aligned
    under the panel title, so they never reach into the column to the left)."""
    case = CASE_L
    epoch = case['final_epoch']
    p_results = load_probe_scores_all_datasets(case['run'], epoch)
    b_results = load_probe_scores_all_datasets(case['baseline'], epoch)
    missing = [d for d in TRANSFER_DATASETS if d not in p_results or d not in b_results]
    if missing:
        raise ValueError(f'case L epoch {epoch}: missing {missing}')
    datasets = list(TRANSFER_DATASETS)
    gaps = {ds: 100 * (p_results[ds]['score'] - b_results[ds]['score'])
            for ds in datasets}

    fine = sorted((d for d in datasets if d in FINE_GRAINED), key=gaps.get)
    other = sorted((d for d in datasets if d not in FINE_GRAINED), key=gaps.get)
    fg = [gaps[d] for d in fine]
    ot = [gaps[d] for d in other]
    groups = [(f'Fine-grained, {len(fine)} datasets', fine),
              (f'Other {len(ot)} datasets', other)]
    rows = []
    for gi, (head, members) in enumerate(groups):
        rows += [head] + members
    y = np.arange(len(rows))
    labels = []
    for yi, row in zip(y, rows):
        if row is None:
            labels.append('')
        elif row in gaps:
            ax.barh(yi, gaps[row], height=0.7, color=S.ONSET_COLORS['early'],
                    linewidth=0)
            labels.append(DATASET_NAMES[row])
        else:
            labels.append('')
            ax.text(head_x, yi + 0.15, row, transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=S.FS_NOTE,
                    color=S.NOTE_COLOR, fontstyle='italic')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.axvline(0, color=S.BASELINE_COLOR, lw=0.6, zorder=3)
    xlim, xticks = _fit_lim('panel D, x', (-19, 3.2), [-16, -8, 0], list(gaps.values()))
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel('Gap to baseline (pp),\nlinear-probe top-1')
    ax.spines['left'].set_visible(False)
    return np.mean(fg), np.mean(ot)


# Figure

def _top_key(fig, top):
    H = fig.get_figheight()
    base = [Line2D([], [], color=S.BASELINE_COLOR, lw=0.8,
                   label='unperturbed baseline (retained 1, gap 0)')]
    onset = S.onset_handles(with_baseline=False, star=True)
    kw = dict(frameon=False, handlelength=2.0, handletextpad=0.5,
              columnspacing=1.3, borderaxespad=0, fontsize=S.FS_LEGEND,
              markerscale=1.2)
    fig.legend(handles=base + onset, loc='center', ncol=4,
               bbox_to_anchor=(0.5, 1 - top / H), **kw)
    fig.legend(handles=S.pert_handles(lines=True), loc='center', ncol=3,
               bbox_to_anchor=(0.5, 1 - (top + 0.17) / H), **kw)


def _partial_notes():
    """One line per appendix panel that leaves out points whose inputs are not computed yet."""
    notes = []
    for panel, (pts, left_out), n, unit in (
            ('A', _final_epoch_points(), len(PERTURBED_RUNS), 'runs'),
            ('B', _distance_gap_points(), 3 * len(PERTURBED_RUNS), 'points')):
        if left_out:
            notes.append(f'Seed {SEED_SUFFIX.removeprefix("_seed")}, partial. {panel}: '
                         f'{len(pts)} of {n} {unit} shown; '
                         + '; '.join(f'{what} not yet computed for {k} {unit}'
                                     for what, k in left_out.items()) + '.')
    return notes


# Main figure: A (left column, with its key beneath) beside D (right column, from the
# top). Appendix figure: the final-epoch retention (A, was B) and gap-vs-distance (B,
# was C) scatters.
H_MAIN = 3.10
COL_R = 3.40


def main_figure():
    W, H = S.TEXT_WIDTH, H_MAIN
    fig = plt.figure(figsize=(W, H))

    # A: T above K, sharing one step axis + one epoch axis (without sub-epoch data,
    # one epoch axis across the whole width).
    a_head = 0.13
    left, w_s, w_e, gap_se, ax_h = 0.55, 1.70, 0.74, 0.14, 0.60
    tops = [0.37, 1.18]
    pairs = []
    for case, top in zip((CASE_T, CASE_K), tops):
        if SUB_EPOCH:
            ax_s = _add_axes(fig, left, top, w_s, ax_h)
            ax_e = _add_axes(fig, left + w_s + gap_se, top, w_e, ax_h)
            pairs.append((ax_s, ax_e))
        else:
            pairs.append((_add_axes(fig, left, top, w_s + gap_se + w_e, ax_h),))
        _case_tag_fig(fig, left + 0.06, top - 0.11, case['label'])
    end_first, transfer_final = panel_a(pairs)
    for ax in pairs[0]:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)
    pairs[0][0].set_ylabel('')
    _fig_text(fig, 0.10, (tops[0] + tops[1] + ax_h) / 2,
              'Retained fraction (perturbed / baseline)', rotation=90,
              ha='center', va='center', fontsize=S.FS_LABEL)
    _header(fig, 0.04, a_head, 'A',
            f'Retained fraction of {"four" if SUB_EPOCH else "two"} readouts, '
            f'onset epoch {CASE_T["onset"]}')

    # A's key, one column under it: the readouts, then the baseline line.
    handles = [Line2D([], [], color=st['color'], lw=S.LW_DATA,
                      marker=st['marker'], markersize=st['ms'],
                      markerfacecolor=st['color'],
                      markeredgecolor=st.get('edge', st['color']),
                      label=st['label']) for st in READOUT_STYLES.values()]
    handles.append(Line2D([], [], color=S.BASELINE_COLOR, lw=0.8,
                          label='unperturbed baseline (retained 1, gap 0)'))
    key_top = tops[1] + ax_h + 0.40
    fig.legend(handles=handles, loc='upper left', ncol=1, frameon=False,
               bbox_to_anchor=(left / W, 1 - key_top / H), handlelength=2.2,
               handletextpad=0.5, fontsize=S.FS_LEGEND, borderaxespad=0,
               labelspacing=0.25)
    if not SUB_EPOCH:
        _fig_text(fig, COL_R, H - 0.05, f'Seed {SEED_SUFFIX.removeprefix("_seed")}: '
                  'epoch resolution; no sub-epoch or ImageNet-probe readouts',
                  fontsize=S.FS_NOTE, color=S.NOTE_COLOR, ha='left', va='bottom')

    # D: the right column from the top.
    d_head = a_head
    d_left, d_w = 4.47, 0.95
    d_top = d_head + 0.33
    d_h = H - 0.42 - d_top
    ax_d = _add_axes(fig, d_left, d_top, d_w, d_h)
    fine_mean, other_mean = panel_d(ax_d, (COL_R + 0.05 - d_left) / d_w)
    x1 = _header(fig, COL_R - 0.02, d_head, 'B',
                 f'Per-dataset gap, final epoch {CASE_L["final_epoch"]}')
    _case_tag_fig(fig, x1 + 0.07, d_head + 0.115, CASE_L['label'])
    S.save(fig, FIGURES, STEM)
    plt.close(fig)
    return end_first, transfer_final, fine_mean, other_mean


def appendix_figure():
    notes = _partial_notes() if PARTIAL else []
    base_h = 2.45
    W, H = S.TEXT_WIDTH, base_h + (0.05 + 0.12 * len(notes) if notes else 0)
    fig = plt.figure(figsize=(W, H))
    _top_key(fig, 0.10)
    head, top, h = 0.62, 0.78, 1.12
    ax_b = _add_axes(fig, 0.72, top, 1.75, h)
    pts_b, r_clip, sup_range = panel_b(ax_b)
    _header(fig, 0.04, head, 'A', 'Final-epoch retention')
    ax_c = _add_axes(fig, 3.55, top, 1.85, h)
    pts_c, rho = panel_c(ax_c)
    _header(fig, 2.85, head, 'B', 'Gap vs. distance to baseline')
    for i, note in enumerate(notes):
        _fig_text(fig, 0.04, base_h + 0.10 + 0.12 * i, note, fontsize=S.FS_NOTE,
                  color=S.NOTE_COLOR, ha='left', va='baseline')
    S.save(fig, FIGURES, STEM_APPENDIX)
    plt.close(fig)
    return pts_b, r_clip, sup_range, pts_c, rho


def make_figure():
    end_first, transfer_final, fine_mean, other_mean = main_figure()
    pts_b, r_clip, sup_range, pts_c, rho = appendix_figure()
    print(dict(end_first=np.round(end_first, 4), transfer_final=np.round(transfer_final, 4),
               r_clip=round(r_clip, 3), sup_range=np.round(sup_range, 4),
               rho=round(rho, 3), n_b=len(pts_b), n_c=len(pts_c),
               n_c_runs=len({q['run'] for q in pts_c}),
               fine_mean=round(fine_mean, 2), other_mean=round(other_mean, 2)))
    for run_name, epoch, why in ABSENT:
        print(f'absent: {run_name} epoch {epoch}: {why}')


def main():
    global SKIP_PARTIAL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--skip-partial', action='store_true',
                        help='leave out checkpoints whose transfer suite is only '
                             'partly probed (listed as absent) instead of failing')
    SKIP_PARTIAL = parser.parse_args().skip_partial
    make_figure()


if __name__ == '__main__':
    main()
