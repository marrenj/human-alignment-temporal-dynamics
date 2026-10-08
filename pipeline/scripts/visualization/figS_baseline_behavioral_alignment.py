"""Supplementary figure: baseline behavioral (human) alignment vs. % of training,
one line per architecture (ResNet50, ViT-B/16, ConvNeXt-B, CLIP-ViT-B/32).

Metric: per-epoch Spearman rho between each model's RDM and the 66-d human
SPoSE embeddings over all 1854 THINGS images (evaluation/run_behavioral_rsa.py,
`<results_root>/posthoc_rdm1854/`). Unlike val-loss curves, this is one consistent metric/scale across all four
architectures, so it's a single overlaid panel rather than 2x2 -- which is
exactly why the x-axis matters here: on a shared panel, a raw-epoch axis
would make CLIP-ViT-B/32's line visually stop dead around epoch 32 next to
three lines still running to 90, implying an unfinished run rather than a
shorter schedule (32 vs. 90 epochs; see each architecture's hyperparameter
table). Plotting % of training (epoch / (total scheduled epochs - 1)) instead
lets all four lines span the same 0-100% range on a like-for-like basis.

ViT-B/16's baseline was evaluated through epoch 151, but is capped at epoch 89
to match the checkpoint-retention range of the other three timm architectures.

Each baseline is scored with the preprocessing its architecture was trained under (the
corrected-preprocessing summaries, `arch_runs.rdm1854_corrected_suffix`), ResNet-50 after
BatchNorm re-estimation and CLIP ViT-B/32 on its class-token features -- the files every other
behavioral-alignment figure reads -- never the untagged legacy-preprocessing summaries.

Writes Fig1b_baseline_alignment{seed suffix}.{pdf,png} to <figures_root>.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_runs import (CLIP_PREFIX, FIGURES as OUTDIR, PARTIAL, RESULTS, SEED_SUFFIX,  # noqa: E402
                       baseline_run, pending, rdm1854_corrected_suffix)

# (run prefix, label, color, max epoch to plot [None = no cap])
ARCHES = [
    ('resnet50_imagenet',   'RN50',          '#5B7FA5', None),
    ('vitb16_imagenet',     'ViT-B/16',      '#E8873C', 89),
    ('convnext_b_imagenet', 'ConvNeXt-B',    '#4FA86E', None),
    (CLIP_PREFIX,         'CLIP-ViT-B/32', '#8E5EAD', None),
]

# Nominal full-schedule length per architecture.
TOTAL_EPOCHS = {
    'RN50': 90,
    'ViT-B/16': 90,
    'ConvNeXt-B': 90,
    'CLIP-ViT-B/32': 32,
}


def pct_of_training(epochs, label):
    total = TOTAL_EPOCHS[label]
    return [100.0 * e / (total - 1) for e in epochs]


def load_traj(prefix, max_epoch=None):
    """(epochs, rho) sorted by epoch for a run's posthoc_rdm1854 baseline, or (None, None)."""
    suffix = ('_bn_recal' if prefix == 'resnet50_imagenet' else '') + rdm1854_corrected_suffix(prefix)
    p = (RESULTS / 'posthoc_rdm1854' / baseline_run(prefix)
         / f'test_things_behavioral_full1854_rsa_summary{suffix}.json')
    if not p.is_file():
        print(f'  not found: {p}')
        return None, None
    with p.open() as f:
        d = json.load(f)
    by_epoch = {}
    for r in d.get('results', []):
        try:
            ep = int(r['epoch'])
        except (KeyError, TypeError, ValueError):
            continue
        score = r.get('rsa_results', {}).get('behavioral_rdm', {}).get('score')
        if score is None or (max_epoch is not None and ep > max_epoch):
            continue
        by_epoch[ep] = float(score)
    epochs = sorted(by_epoch)
    return epochs, [by_epoch[e] for e in epochs]


def make_figure():
    fig, ax = plt.subplots(figsize=(4.5, 3.4))

    absent, notes = [], []
    for prefix, label, color, max_epoch in ARCHES:
        epochs, rhos = load_traj(prefix, max_epoch)
        if not epochs:
            print(f'  {label}: MISSING')
            if PARTIAL:
                pending(f'{baseline_run(prefix)} behavioral RSA summary')
                absent.append(label)
            continue
        gaps = sorted(set(range(TOTAL_EPOCHS[label])) - set(epochs))
        if gaps:
            print(f'  WARNING: {label} has no score at epochs {gaps}')
        pct = pct_of_training(epochs, label)
        if PARTIAL and gaps:
            pending(f'{baseline_run(prefix)} behavioral RSA epochs {gaps}')
            notes.append(f'{label}: {len(epochs)} of {TOTAL_EPOCHS[label]} epochs')
            # A NaN between non-consecutive epochs, so no segment bridges a gap.
            xs, ys = [], []
            for i, e in enumerate(epochs):
                if i and e - epochs[i - 1] > 1:
                    xs.append(float('nan'))
                    ys.append(float('nan'))
                xs.append(pct[i])
                ys.append(rhos[i])
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label, zorder=3,
                    marker='o', markersize=2)
        else:
            ax.plot(pct, rhos, color=color, linewidth=1.5, label=label, zorder=3)
        print(f'  {label}: {len(epochs)} epochs [{epochs[0]}-{epochs[-1]}] '
              f'({pct[0]:.0f}-{pct[-1]:.0f}% of {TOTAL_EPOCHS[label]}), '
              f'rho {rhos[0]:.3f} -> {rhos[-1]:.3f} (peak {max(rhos):.3f})')

    ax.set_xlabel('% of training')
    ax.set_ylabel(r'Behavioral alignment vs. human (Spearman $\rho$)')
    if PARTIAL and not ax.lines:
        ax.text(0.5, 0.5, 'not yet computed', transform=ax.transAxes, ha='center',
                va='center', fontsize=7, color='#555555')
    else:
        # Lower right covers the line ends that fall there; a partial render moves
        # the legend off the data rather than hide what has been computed.
        ax.legend(loc='best' if PARTIAL else 'lower right', framealpha=0.9,
                  edgecolor='none', handlelength=1.4)

    if absent:
        notes.insert(0, 'not yet computed: ' + ', '.join(absent))
    if notes:
        fig.text(0.01, 0.01, f'Seed {SEED_SUFFIX.removeprefix("_seed")}, partial; '
                 + '; '.join(notes) + '.', ha='left', va='bottom', fontsize=6.5,
                 color='#555555')
    fig.tight_layout(rect=[0, 0.05 if notes else 0, 1, 0.98])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = OUTDIR / f'Fig1b_baseline_alignment{SEED_SUFFIX}.{ext}'
        fig.savefig(out)
        print(f'Saved {out}')
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print(f'Reading posthoc_rdm1854 baselines from: {RESULTS}')
    make_figure()


if __name__ == '__main__':
    main()
