"""Per-ROI change in brain alignment at the perturbed epoch, THINGS-fMRI and NOD (appendix figures).

Every perturbed run (onsets early / mid / late = 5 / 30 / 70;
CLIP ViT-B/32 5 / 13 / 28), scored with the corrected-preprocessing FR-RSA files
that Fig 2E reads (ResNet-50 BatchNorm-recalibrated), against its baseline at
exactly the same epoch.

Small multiples: rows = perturbation type, columns = architecture; x = onset
epoch, y = perturbed - baseline in-sample FR-RSA Spearman rho, one line per ROI.
Drawn at the paper's text width with the shared `paper_style` grammar.

Writes Fig_neural_critical_window_by_roi_{things,nod}_canonical.{pdf,png} to
<figures_root> and prints the numbers behind the NOD caption's comparison with THINGS.

Usage:
    python pipeline/scripts/visualization/figS_neural_alignment_by_roi.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fig2_test_performance_defs_in_caption as F2
import make_fig7_summary_defs_in_caption as F7
from arch_runs import FIGURES as OUTDIR, frrsa_dir
from figS_baseline_neural_alignment import (BILATERAL_LOC_COLOR, FAMILIES,
                                           FAMILY_COLOR, NOD_COLOR, VA_COLORS,
                                           VA_HIER, nod_roi_style, parse_roi,
                                           roi_style)
import frrsa_fig_data as D
# The modules above set their own rcParams at import; the print-size style
# has to be applied after them.
import paper_style as S
S.apply()

ARCHES, TIMINGS, ONSET_EPOCH = F2.ARCHES, F2.TIMINGS, F2.ONSET_EPOCH
PERTS = ['image_noise', 'uniform_images', 'label_shuffle']
N_ROIS = {'things': 27, 'nod': 14}

W = S.TEXT_WIDTH
LEFT, TOP, AX_W, AX_H, COL_PITCH, ROW_PITCH = 0.62, 0.22, 0.74, 0.70, 0.86, 0.80
KEY_X = LEFT + 3 * COL_PITCH + AX_W + 0.22
H = TOP + 2 * ROW_PITCH + AX_H + 0.42


def _scores(arch_key, run, dataset, epoch):
    """{roi: in-sample Spearman rho} from the corrected-preprocessing FR-RSA file."""
    path = (frrsa_dir(run)
            / f'inference_results_{dataset}_frrsa_{F7.BRAIN_TAG[arch_key]}_epoch{epoch}.json')
    with path.open() as f:
        blocks = json.load(f)
    scores = {roi: b['insample_score'] for roi, b in blocks.items()}
    if len(scores) != N_ROIS[dataset] or None in scores.values():
        raise ValueError(f'{path}: expected {N_ROIS[dataset]} ROI scores')
    return scores


def load_deltas(dataset):
    """{(arch, pert): {roi: [delta at early, mid, late onset]}}, perturbed minus
    baseline at the same onset epoch."""
    out = {}
    for arch_key, _, _ in ARCHES:
        base = F2._baseline_name(arch_key)
        for pert in PERTS:
            series = {}
            for timing in TIMINGS:
                epoch = ONSET_EPOCH[arch_key][timing]
                p = _scores(arch_key, F2._run_name(arch_key, pert, timing), dataset, epoch)
                b = _scores(arch_key, base, dataset, epoch)
                for roi in p:
                    series.setdefault(roi, []).append(p[roi] - b[roi])
            out[(arch_key, pert)] = {roi: np.array(v) for roi, v in series.items()}
    return out


def _axes(fig, left, top, width, height):
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def _things_key(fig):
    cax = _axes(fig, KEY_X, TOP + 0.05, 0.07, 1.05)
    sm = cm.ScalarMappable(cmap=ListedColormap(VA_COLORS),
                           norm=BoundaryNorm(np.arange(len(VA_HIER) + 1) - 0.5, len(VA_HIER)))
    cb = fig.colorbar(sm, cax=cax, ticks=range(len(VA_HIER)))
    cb.ax.set_yticklabels(VA_HIER, fontsize=S.FS_NOTE)
    cb.ax.invert_yaxis()
    cb.ax.tick_params(length=0, pad=1.5)
    cb.outline.set_linewidth(0.5)
    fig.text((KEY_X - 0.05) / W, 1 - (TOP + 0.02) / H, 'Retinotopic areas', fontsize=S.FS_NOTE,
             ha='left', va='bottom')
    handles = [Line2D([], [], color=FAMILY_COLOR[f], lw=1.6, label=f) for f in FAMILIES]
    handles.append(Line2D([], [], color=BILATERAL_LOC_COLOR, lw=1.6, ls=(0, (1, 1)),
                          label='LOC (bilateral)'))
    handles.append(Line2D([], [], color='#444444', lw=1.2, ls='-', label='left hemisphere'))
    handles.append(Line2D([], [], color='#444444', lw=1.2, ls=(0, (4, 1.5)),
                          label='right hemisphere'))
    fig.legend(handles=handles, loc='upper left',
               bbox_to_anchor=((KEY_X - 0.06) / W, 1 - (TOP + 1.22) / H),
               frameon=False, fontsize=S.FS_NOTE, handlelength=1.6, handletextpad=0.4,
               labelspacing=0.25, borderaxespad=0, borderpad=0)


def _nod_key(fig):
    handles = [Line2D([], [], color=NOD_COLOR[roi], lw=1.6, label=roi) for roi in D.NOD_ROI_ORDER]
    fig.legend(handles=handles, loc='upper left',
               bbox_to_anchor=((KEY_X - 0.06) / W, 1 - (TOP + 0.02) / H),
               frameon=False, fontsize=S.FS_NOTE, handlelength=1.6, handletextpad=0.4,
               labelspacing=0.25, borderaxespad=0, borderpad=0,
               title='ROI, early to late', title_fontsize=S.FS_NOTE)


def make_figure(dataset, deltas, style, key):
    allv = np.concatenate([v for s in deltas.values() for v in s.values()])
    lo, hi = min(allv.min(), 0.0), max(allv.max(), 0.0)
    pad = 0.08 * (hi - lo)
    fig = plt.figure(figsize=(W, H))
    for i, pert in enumerate(PERTS):
        for j, (arch_key, arch_label, _) in enumerate(ARCHES):
            ax = _axes(fig, LEFT + j * COL_PITCH, TOP + i * ROW_PITCH, AX_W, AX_H)
            ax.axhline(0, color=S.REF_COLOR, lw=S.LW_REF, ls=(0, (3, 2)), zorder=1)
            series = deltas[(arch_key, pert)]
            for roi in sorted(series, key=lambda r: (parse_roi(r)[0] != 'va', r)
                              if dataset == 'things' else D.NOD_ROI_ORDER.index(r)):
                color, ls, lw, alpha, z = style(roi)
                ax.plot(range(len(TIMINGS)), series[roi], color=color, ls=ls, lw=lw * 0.8,
                        alpha=alpha, zorder=z, marker='o', ms=1.8, mew=0)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.3, len(TIMINGS) - 0.7)
            ax.set_xticks(range(len(TIMINGS)))
            ax.set_xticklabels([str(ONSET_EPOCH[arch_key][t]) for t in TIMINGS]
                               if i == len(PERTS) - 1 else [])
            for tick, t in zip(ax.get_xticklabels(), TIMINGS):
                tick.set_color(S.ONSET_COLORS[t])
            if j:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(S.PERT_LABEL[pert], fontsize=S.FS_LABEL)
            if i == 0:
                ax.set_title(arch_label, fontsize=S.FS_LABEL, pad=3)
    fig.text(0.02 / W, 1 - (TOP + ROW_PITCH + AX_H / 2) / H, r'$\Delta$ brain alignment ($\rho$)',
             rotation=90, ha='left', va='center', fontsize=S.FS_LABEL)
    fig.text((LEFT + (3 * COL_PITCH + AX_W) / 2) / W, 0.03 / H, 'Onset epoch',
             ha='center', va='bottom', fontsize=S.FS_LABEL)
    key(fig)
    S.save(fig, OUTDIR, f'Fig_neural_critical_window_by_roi_{dataset}_canonical')
    plt.close(fig)


def report(things, nod):
    """The NOD caption compares NOD with THINGS; print what that comparison rests on."""
    def cell_means(d):
        return np.array([np.mean([v[k] for v in s.values()])
                         for _, s in sorted(d.items()) for k in range(len(TIMINGS))])
    t_cells, n_cells = cell_means(things), cell_means(nod)
    print(f'mean |delta| over ROIs and runs: THINGS {np.mean(np.abs(t_cells)):.4f}, '
          f'NOD {np.mean(np.abs(n_cells)):.4f} (ratio {np.mean(np.abs(n_cells)) / np.mean(np.abs(t_cells)):.2f})')
    print(f'Pearson r of per-run ROI-mean delta, THINGS vs NOD (36 runs): '
          f'{np.corrcoef(t_cells, n_cells)[0, 1]:.2f}')
    print('NOD per-ROI mean delta over all 36 runs:')
    for roi in D.NOD_ROI_ORDER:
        print(f'  {roi:5s} {np.mean([s[roi].mean() for s in nod.values()]):+.4f}')


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    things, nod = load_deltas('things'), load_deltas('nod')
    make_figure('things', things, roi_style, _things_key)
    make_figure('nod', nod, nod_roi_style, _nod_key)
    report(things, nod)


if __name__ == '__main__':
    main()
