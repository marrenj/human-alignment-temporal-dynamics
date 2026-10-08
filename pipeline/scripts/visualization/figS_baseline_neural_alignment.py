"""Supplementary figure: per-ROI baseline neural-alignment (FR-RSA) trajectories.

ONE line per ROI, split into 4 panels (one per architecture). Shows how different
brain regions' alignment develops over training within each architecture. One
figure per dataset (THINGS-fMRI, NOD).

Reads the corrected-preprocessing FR-RSA files (`frrsa_fig_data`, `corrected=True`),
ResNet-50 BatchNorm-recalibrated, not the legacy untagged ones.

THINGS-fMRI (27 ROIs) has two ROI families:
  - Retinotopic pRF visual areas (neuropythy/Benson `varea`, 12 areas). The raw
    files are named va-1..va-12, but that INDEX IS AN ATLAS LABEL, NOT A HIERARCHY
    RANK -- va-11/va-12 are V3b/V3a (dorsal, ~V3 level). We therefore name each by
    region and color by an approximate hierarchical order (V1 -> TO2), with dorsal
    V3a/V3b grouped near V3. Mapping verified against the THINGS-fMRI repo's
    get_prf_rois() (MRI/thingsmri/utils.py).
  - Category-selective regions (EBA/FFA/OFA/LOC/PPA/RSC/TOS, +bilateral LOC):
    distinct warm colors; left hemisphere = solid, right = dashed.

Depth caveat as elsewhere: absolute FR-RSA levels are in-sample (positively biased
with layer count); read within-architecture / within-ROI shape, not raw magnitude.

Writes Fig_neural_trajectories_by_roi_{things,nod}{seed suffix}.{pdf,png} to <figures_root>.

Usage:
    python pipeline/scripts/visualization/figS_baseline_neural_alignment.py [--raw-epochs]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frrsa_fig_data as D
from arch_runs import FIGURES as OUTDIR, PARTIAL, SEED_SUFFIX, baseline_run, pending

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
})

DATASET = 'things'
# Percent of training rather than raw epoch, so CLIP-ViT-B/32's 32-epoch
# schedule is comparable to the others' 90 instead of looking truncated
# (frrsa_fig_data.pct_of_training). `--raw-epochs` gives the raw-epoch axis.
X_AS_PCT = True

# --- retinotopic pRF areas: va-N -> region (verified vs THINGS-fMRI get_prf_rois) ---
VA_NAME = {
    'va-1': 'V1', 'va-2': 'V2', 'va-3': 'V3', 'va-4': 'hV4',
    'va-5': 'VO1', 'va-6': 'VO2', 'va-7': 'LO1', 'va-8': 'LO2',
    'va-9': 'TO1', 'va-10': 'TO2', 'va-11': 'V3b', 'va-12': 'V3a',
}
# approximate hierarchical order (NOT the atlas index); dorsal V3a/V3b grouped near V3
VA_HIER = ['V1', 'V2', 'V3', 'V3a', 'V3b', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1', 'TO2']
VA_RANK = {n: i for i, n in enumerate(VA_HIER)}
_BLUES = matplotlib.colormaps['Blues']
VA_COLORS = [_BLUES(x) for x in np.linspace(0.32, 1.0, len(VA_HIER))]  # light->dark by hierarchy

# --- category-selective ROIs -------------------------------------------------------
FAMILIES = ['EBA', 'FFA', 'OFA', 'LOC', 'PPA', 'RSC', 'TOS']
FAMILY_COLOR = {
    'EBA': '#E8873C', 'FFA': '#C44E52', 'OFA': '#E377C2', 'LOC': '#8C564B',
    'PPA': '#9467BD', 'RSC': '#D4A017', 'TOS': '#2CA02C',
}
BILATERAL_LOC_COLOR = '#000000'


def parse_roi(roi):
    if roi.startswith('va-'):
        return ('va', roi)
    if roi == 'LOC':
        return ('cat_bilat', 'LOC')
    if roi[0] in 'lr' and roi[1:] in FAMILIES:
        return ('cat', roi[1:], roi[0])
    return ('other', roi)


def roi_style(roi):
    """(color, linestyle, linewidth, alpha, zorder) for an ROI."""
    kind = parse_roi(roi)
    if kind[0] == 'va':
        name = VA_NAME[kind[1]]
        return VA_COLORS[VA_RANK[name]], '-', 1.2, 0.85, 2
    if kind[0] == 'cat':
        ls = '-' if kind[2] == 'l' else (0, (4, 1.5))
        return FAMILY_COLOR[kind[1]], ls, 1.5, 0.95, 3
    if kind[0] == 'cat_bilat':
        return BILATERAL_LOC_COLOR, (0, (1, 1)), 1.5, 0.95, 3
    return '#BBBBBB', '-', 1.0, 0.6, 1


# --- NOD: 14 ROIs, already a readable count, so all of them are drawn ---------
# Colored by position along the V1 -> high-level ordering rather than by family:
# NOD's parcellation is retinotopic/object-selective throughout and has no
# left/right split, so the THINGS hemisphere-linestyle vocabulary does not apply.
# Defined here so both NOD figures take their colors from one place --
# figS_neural_alignment_by_roi already imports styling from this module.
_NOD_CMAP = matplotlib.colormaps['viridis']
NOD_COLOR = {roi: _NOD_CMAP(x) for roi, x in
             zip(D.NOD_ROI_ORDER, np.linspace(0.0, 0.9, len(D.NOD_ROI_ORDER)))}


def nod_roi_style(roi):
    """(color, linestyle, linewidth, alpha, zorder) for a NOD ROI."""
    return NOD_COLOR.get(roi, '#BBBBBB'), '-', 1.4, 0.95, 2


# Per dataset: which ROIs to draw (raw key, legend label), how to style them,
# and the legend title. THINGS draws the 8-ROI highlight subset so it pairs with
# Fig_neural_critical_window_highlighted_things; NOD draws all 14 so it pairs
# with Fig_neural_critical_window_by_roi_nod. In both cases the trajectory
# figure and its critical-window counterpart show the same ROIs in the same
# colors, which is the point -- they are meant to be read together.
DATASET_SPEC = {
    'things': dict(
        rois=lambda: D.THINGS_HIGHLIGHT_ROIS,
        style=roi_style,
        legend_title='ROI',
    ),
    'nod': dict(
        rois=lambda: [(roi, roi) for roi in D.NOD_ROI_ORDER],
        style=nod_roi_style,
        legend_title='ROI (V1 → late)',
    ),
}


def fit_ylim(axes):
    # `set_ylim(bottom=0)` freezes the shared axis at the first panel's autoscaled top,
    # so a later panel's peak, or a score below 0, can fall outside it; widen it by a 5%
    # margin on whichever side the data overflow.
    ys = np.concatenate([line.get_ydata() for ax in axes for line in ax.lines]).astype(float)
    (lo, hi), dlo, dhi = axes[0].get_ylim(), np.nanmin(ys), np.nanmax(ys)
    if dlo < lo or dhi > hi:
        pad = 0.05 * (max(hi, dhi) - min(lo, dlo))
        axes[0].set_ylim(dlo - pad if dlo < lo else lo, dhi + pad if dhi > hi else hi)


def check_baseline_grid():
    """Fail naming every epoch missing from the baselines' regular FR-RSA grid
    (in partial mode, record them as pending instead)."""
    gaps = []
    for ds_key, _ in D.DATASETS:
        for prefix, arch_label, _ in D.ARCHES:
            eps = D.missing_baseline_epochs(prefix, ds_key, corrected=True)
            if eps and PARTIAL:
                pending(f'{baseline_run(prefix)} {ds_key} FR-RSA epochs {eps}')
            elif eps:
                gaps.append(f'  {arch_label}, {ds_key}: epochs {eps}\n'
                            f'    e.g. {D._json_path(prefix, "baseline", ds_key, eps[0], corrected=True)}')
    if gaps:
        raise FileNotFoundError('baseline FR-RSA results missing:\n' + '\n'.join(gaps))


def break_at_gaps(eps, ys, missing):
    """(eps, ys) with a NaN between two epochs that have a missing grid epoch
    between them, so no line segment bridges the gap."""
    xs_out, ys_out = [], []
    for i, (e, y) in enumerate(zip(eps, ys)):
        if i and any(eps[i - 1] < m < e for m in missing):
            xs_out.append(np.nan)
            ys_out.append(np.nan)
        xs_out.append(e)
        ys_out.append(y)
    return np.array(xs_out, float), np.array(ys_out, float)


def make_figure(ds_key):
    spec = DATASET_SPEC[ds_key]
    rois, style = spec['rois'](), spec['style']

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=False, sharey=True)
    axes = axes.flatten()

    for ax, (prefix, arch_label, _) in zip(axes, D.ARCHES):
        tr = D.baseline_trajectory(prefix, ds_key, corrected=True)
        per_roi = tr['per_roi']
        missing = D.missing_baseline_epochs(prefix, ds_key, corrected=True) if PARTIAL else []
        if not per_roi and not PARTIAL:
            ax.set_title(f'{arch_label} (no data)', loc='left')
            continue
        for roi, _ in rois:
            if roi not in per_roi:
                continue
            pts = per_roi[roi]
            eps = np.array(sorted(pts.keys()))
            ys = np.array([pts[e] for e in eps], float)
            color, ls, lw, alpha, z = style(roi)
            if missing:
                eps, ys = break_at_gaps(eps, ys, missing)
            # A raw-epoch axis makes CLIP-ViT-B/32's 32-epoch schedule look like
            # a truncated 90-epoch one; percent of training puts all four panels
            # on the same footing.
            xs = D.pct_of_training(prefix, eps) if X_AS_PCT else eps
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw,
                    alpha=alpha, zorder=z, solid_capstyle='round',
                    **(dict(marker='o', markersize=2.5) if missing else {}))
        ax.set_title(arch_label, loc='left')
        ax.set_xlabel('% of training completed' if X_AS_PCT else 'Training epoch')
        ax.set_ylabel(r'LR-RSA $\rho$ (per ROI)')
        # An empty panel would freeze the shared y-axis at its default (0, 1).
        if ax.lines:
            ax.set_ylim(bottom=0)
        if missing:
            # The full schedule, so a lone point sits where it belongs in training.
            hi = 100 if X_AS_PCT else D.TOTAL_EPOCHS[prefix] - 1
            ax.set_xlim(-0.02 * hi, 1.02 * hi)
            n_grid = len(D.baseline_grid(prefix))
            ax.set_title(f'Seed {SEED_SUFFIX.removeprefix("_seed")}, partial: '
                         f'{n_grid - len(missing)} of {n_grid} epochs',
                         loc='right', fontsize=6.5, fontweight='normal', color='#555555')
            if not per_roi:
                ax.text(0.5, 0.5, 'not yet computed', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, color='#555555')
        else:
            ax.margins(x=0.02)
        ax.tick_params(length=2.5)
    if any(ax.lines for ax in axes):
        fit_ylim(axes)

    fig.subplots_adjust(right=0.79, wspace=0.22, hspace=0.42)

    # One compact legend, same entries in the same order and colors as this
    # dataset's critical-window figure. Two columns for NOD, whose 14 entries
    # would otherwise run taller than the plot area.
    handles = [Line2D([0], [0], color=style(roi)[0], linestyle=style(roi)[1],
                      linewidth=2.0, label=label) for roi, label in rois]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.815, 0.5),
               frameon=False, handlelength=1.6, labelspacing=0.55,
               ncol=2 if len(handles) > 10 else 1, columnspacing=1.0,
               title=spec['legend_title'], title_fontsize=7.5)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = OUTDIR / f'Fig_neural_trajectories_by_roi_{ds_key}{SEED_SUFFIX}.{ext}'
        fig.savefig(out)
        print(f'Saved {out}')
    plt.close(fig)


def main():
    global X_AS_PCT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--raw-epochs', action='store_true',
                    help='x-axis in raw training epochs instead of %% of training')
    X_AS_PCT = not ap.parse_args().raw_epochs
    check_baseline_grid()
    for ds_key, _ in D.DATASETS:
        make_figure(ds_key)


if __name__ == '__main__':
    main()
