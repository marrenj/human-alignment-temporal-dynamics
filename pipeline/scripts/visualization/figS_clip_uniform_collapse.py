"""Supplementary figure: CLIP ViT-B/32's image features after one early epoch of uniform images.

A  PCA of the 768-d CLS features of ImageNet val images (every 10th image
   shown), baseline at epoch 5 and the uniform-images run at the end of its
   perturbed epoch (5) and at the last epoch (31), coloured by image luminance.
B  Spearman correlation of PC1 with twelve low-level image statistics, for the
   collapsed and the baseline features.
C  CIFAR-10 linear-probe top-1 on seven feature sets.

Reads `<results_root>/clip_uniform_collapse/{summary.json, pca_points.npz}`
from `pipeline/scripts/analysis/analyze_clip_uniform_collapse.py`
(`clip_uniform_collapse_seed43/` under `TP_SEED_SUFFIX=_seed43`).
Drawn at the paper's text width with the shared `paper_style` grammar.

Usage:
    python pipeline/scripts/visualization/figS_clip_uniform_collapse.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as S
from arch_runs import FIGURES as OUTDIR, RESULTS, SEED_SUFFIX, pending, require_current_probe
S.apply()

DATA_DIR = RESULTS / f'clip_uniform_collapse{SEED_SUFFIX}'
STEM = 'FigS_clip_uniform_collapse'

W, H = S.TEXT_WIDTH, 5.0
A_TOP, ROW2_TOP = 0.30, 2.20
COLLAPSED_COLOR = S.ONSET_COLORS['early']
BASELINE_BAR = '#8a8a8a'
IMAGE_BAR = '#bdbdbd'

PCA_PANELS = [('baseline, epoch 5', 'baseline, epoch 5'),
              ('uniform images, epoch 5', 'uniform images, epoch 5'),
              ('uniform images, epoch 31', 'uniform images, epoch 31')]
PROBE_ROWS = [  # (feature set in summary.json, label, colour)
    ('baseline CLIP features', 'baseline CLIP', S.BASELINE_COLOR),
    ('collapsed CLIP features', 'collapsed CLIP', COLLAPSED_COLOR),
    ('collapsed, arc axis (PC1) only', 'collapsed CLIP,\nPC1 only', COLLAPSED_COLOR),
    ('collapsed, top 3 PCs removed', 'collapsed CLIP,\ntop 3 PCs removed', COLLAPSED_COLOR),
    ('collapsed, low-level stats regressed out', 'collapsed CLIP,\nstatistics regressed out',
     COLLAPSED_COLOR),
    ('raw pixels (32 x 32 x 3)', 'raw pixels', IMAGE_BAR),
    ('12 low-level image statistics', '12 image statistics', IMAGE_BAR),
]


def _axes(fig, left, top, width, height):
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def _not_computed(ax):
    ax.text(0.5, 0.5, 'not yet computed', transform=ax.transAxes, ha='center',
            va='center', fontsize=S.FS_NOTE, color=S.NOTE_COLOR)


def panel_a(fig, points):
    """`points` None: the PCA points are not computed yet (partial mode)."""
    axes = []
    for i, (key, title) in enumerate(PCA_PANELS):
        ax = _axes(fig, 0.40 + i * 1.64, A_TOP, 1.24, 1.24)
        if points is None:
            _not_computed(ax)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            lum = points['luminance']
            lo, hi = np.percentile(lum, [2, 98])
            pc = points[key]
            order = np.random.default_rng(0).permutation(len(pc))
            sc = ax.scatter(pc[order, 0], pc[order, 1], c=lum[order], cmap='viridis',
                            vmin=lo, vmax=hi, s=0.8, linewidths=0, rasterized=True)
            ax.set_aspect('equal', 'datalim')
        ax.set_title(title, fontsize=S.FS_LABEL, pad=3)
        ax.set_xlabel('PC1', labelpad=1)
        ax.set_ylabel('PC2', labelpad=1)
        ax.tick_params(labelsize=S.FS_TICK)
        edge = S.BASELINE_COLOR if key.startswith('baseline') else COLLAPSED_COLOR
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(edge)
            sp.set_linewidth(1.4 if key == 'uniform images, epoch 5' else 0.8)
        axes.append(ax)
    if points is not None:
        cax = _axes(fig, 5.07, A_TOP + 0.10, 0.07, 1.04)
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label('image luminance', fontsize=S.FS_LABEL, labelpad=2)
        cb.ax.tick_params(labelsize=S.FS_TICK)
    S.panel_header(fig, axes, 'A',
                   'PCA of CLIP ViT-B/32 image features, ImageNet validation images')
    return axes


def panel_b(fig, summary, top):
    if summary is None:
        ax = _axes(fig, 1.20, top, 1.40, 2.05)
        _not_computed(ax)
        ax.set_yticks([])
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel('Spearman ρ with PC1')
        return ax
    names = summary['stat_names']
    by_label = {s['label']: s for s in summary['imagenet_states']}
    series = [  # (state, face colour, edge colour, legend label)
        ('uniform images, epoch 5', COLLAPSED_COLOR, COLLAPSED_COLOR, 'uniform images, epoch 5'),
        ('uniform images, epoch 10', 'white', COLLAPSED_COLOR, 'uniform images, epoch 10'),
        ('baseline, epoch 5', BASELINE_BAR, BASELINE_BAR, 'baseline, epoch 5'),
    ]
    ax = _axes(fig, 1.20, top, 1.40, 2.05)
    y = np.arange(len(names))
    for yi in y[::2]:
        ax.axhspan(yi - 0.5, yi + 0.5, color='#f2f2f2', lw=0, zorder=0)
    for k, (label, face, edge, legend) in enumerate(series):
        rho = [by_label[label]['spearman']['PC1'][n] for n in names]
        ax.plot(rho, y + (k - 1) * 0.22, ls='none', marker='o', ms=3.4,
                markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.8,
                label=legend, zorder=3)
    ax.axvline(0, color=S.BASELINE_COLOR, lw=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=S.FS_TICK)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel('Spearman ρ with PC1')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=1, fontsize=S.FS_LEGEND,
              frameon=False, handletextpad=0.3, borderaxespad=0, labelspacing=0.25)
    return ax


def panel_c(fig, summary, top):
    ax = _axes(fig, 4.12, top, 1.23, 2.05)
    y = np.arange(len(PROBE_ROWS))
    if summary is None:
        _not_computed(ax)
    else:
        probes = {p['feature_set']: require_current_probe(p, f"summary.json {p['feature_set']}")
                  for p in summary['cifar10']['probes']}
        for yi, (key, _, color) in zip(y, PROBE_ROWS):
            acc = probes[key]['top1'] * 100
            ax.barh(yi, acc, height=0.62, color=color, zorder=3)
            ax.text(acc + 1.5, yi, f'{acc:.1f}', va='center', ha='left', fontsize=S.FS_NOTE,
                    color=S.NOTE_COLOR)
    ax.axvline(10, color=S.REF_COLOR, lw=S.LW_REF, ls=(0, (2, 2)), zorder=2)
    ax.text(10, 1.01, 'chance', transform=ax.get_xaxis_transform(), fontsize=S.FS_NOTE,
            color=S.NOTE_COLOR, va='bottom', ha='center')
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label, _ in PROBE_ROWS], fontsize=S.FS_TICK,
                       linespacing=1.0)
    ax.set_ylim(len(PROBE_ROWS) - 0.5, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('CIFAR-10 top-1 (%)')
    return ax


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    summary = points = None
    try:
        with (DATA_DIR / 'summary.json').open() as f:
            summary = json.load(f)
    except FileNotFoundError:
        pending(f'{DATA_DIR.name} summary.json (panels B, C)')
    try:
        points = dict(np.load(DATA_DIR / 'pca_points.npz'))
    except FileNotFoundError:
        pending(f'{DATA_DIR.name} pca_points.npz (panel A)')
    fig = plt.figure(figsize=(W, H))
    axes_a = panel_a(fig, points)
    ax_b = panel_b(fig, summary, ROW2_TOP)
    ax_c = panel_c(fig, summary, ROW2_TOP)
    # Without its statistic names, B's own left edge would push its header into C's.
    S.panel_header(fig, ax_b, 'B', 'PC1 vs. low-level image statistics', y_axes=[ax_b, ax_c],
                   x_axes=axes_a[0] if summary is None else None)
    S.panel_header(fig, ax_c, 'C', 'CIFAR-10 linear probe by feature set', y_axes=[ax_b, ax_c])
    if summary is None or points is None:
        missing = [f for f, v in (('pca_points.npz', points), ('summary.json', summary))
                   if v is None]
        fig.text(0.40 / W, 1 - 4.80 / H, f'Seed {SEED_SUFFIX.removeprefix("_seed")}, '
                 f'partial: {DATA_DIR.name}/{" and ".join(missing)} not yet computed.',
                 fontsize=S.FS_NOTE, color=S.NOTE_COLOR, ha='left', va='baseline')
    S.save(fig, OUTDIR, STEM)
    plt.close(fig)
    if summary is None:
        return
    for s in summary['imagenet_states']:
        print(f"{s['label']:26s} PC1 {s['pc_variance_fraction'][0]:.3f} "
              f"eff.dim {s['effective_dim']:6.1f} "
              f"rho(PC1, luminance) {s['spearman']['PC1']['luminance']:+.3f}")
    for p in summary['cifar10']['probes']:
        print(f"CIFAR-10 {p['feature_set']:42s} {p['top1'] * 100:5.1f}%")


if __name__ == '__main__':
    main()
