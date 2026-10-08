"""Supplementary figure: behavioral alignment measured with CKA instead of RSA.

Every behavioral-alignment readout in the paper is the Spearman rho between
model and SPoSE RDMs (RSA). The same scoring runs also computed linear CKA
between the model's features and the 66-d SPoSE embeddings, on the same 1,854
THINGS images, checkpoints and preprocessing (ResNet-50 BatchNorm-recalibrated).
This figure repeats the RSA analyses with CKA:

A  CKA by training epoch (the counterpart of Fig 2A's alignment row).
B  Retained fraction at the onset epoch, CKA vs RSA, one point per run.
C  Gap to baseline at the final epoch, CKA vs RSA (supervised | CLIP).
D  Epochs to regain half the onset drop, CKA vs RSA (Fig 2D's protocol).

It also prints each paper claim about behavioral alignment re-tested with CKA,
including Fig 3's separability / fidelity correlations.

Usage:
    python pipeline/scripts/visualization/figS_cka_behavioral_alignment.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig2_performance_alignment as M
import make_fig7_summary_defs_in_caption as F7
import make_fig_cross_architecture as CA
from arch_runs import (CLIP_PREFIX, FIGURES as OUTDIR, PARTIAL, pending,
                       rdm1854_corrected_suffix, structure_analysis_path)
# The modules above set their own rcParams at import; the print-size style
# has to be applied after them.
import paper_style as S
S.apply()

F2 = M.F2
STEM = 'FigS_cka_behavioral_alignment'
STRUCTURE_JSON = structure_analysis_path()

ARCHES, SUPERVISED, TIMINGS, PERTS = M.ARCHES, M.SUPERVISED, M.TIMINGS, M.PERTS
ONSET_EPOCH, N_EPOCHS = M.ONSET_EPOCH, M.N_EPOCHS


# ------------------------------------------------------------------ data

def cka_curve(arch_key, run):
    """{epoch: linear CKA with SPoSE}, from the same files as the RSA curve."""
    suffix = rdm1854_corrected_suffix(arch_key)
    if arch_key in F7.BN_RECAL_ARCHES:
        suffix = '_bn_recal' + suffix
    traj = CA.load_traj(run, suffix)
    if traj is None:
        return {}
    epochs, _, cka = traj
    return dict(zip(epochs.astype(int).tolist(), cka.tolist()))


def load_cka_curves(rsa):
    """{run: {epoch: CKA}} with exactly the epochs the RSA curve has."""
    out = {}
    for arch_key, _, _ in ARCHES:
        for rn in F2._runs_of(arch_key):
            curve = cka_curve(arch_key, rn)
            missing = sorted(set(rsa[rn]) - set(curve))
            if missing:
                raise ValueError(f'{rn}: RSA but no CKA at epochs {missing[:5]}')
            out[rn] = curve
    return out


def median_epoch_change(curves):
    """Largest, over architectures, of the baseline's median |change| between
    consecutive epochs -- the rule behind Fig 2D's minimum drop, per metric.
    A baseline not yet scored at every epoch (partial mode) is left out."""
    out = []
    for arch_key in complete_baselines(curves):
        c = curves[F2._baseline_name(arch_key)]
        eps = sorted(c)
        out.append(np.median(np.abs(np.diff([c[e] for e in eps]))))
    return max(out)


def complete_baselines(curves):
    """Architectures whose baseline is scored at every epoch."""
    return [arch_key for arch_key, _, n in ARCHES
            if all(e in curves[F2._baseline_name(arch_key)] for e in range(n))]


def per_run(rsa, cka, sub, drops):
    """One row per perturbed run with every quantity the panels compare."""
    rows = []
    for arch_key, _, n in ARCHES:
        base = F2._baseline_name(arch_key)
        for timing in TIMINGS:
            onset = ONSET_EPOCH[arch_key][timing]
            for pert in PERTS:
                rn = F2._run_name(arch_key, pert, timing)
                row = dict(arch=arch_key, pert=pert, timing=timing)
                for name, curves in (('rsa', rsa), ('cka', cka)):
                    # None: not scored yet (partial mode).
                    on = curves[rn].get(onset), curves[base].get(onset)
                    fin = curves[rn].get(n - 1), curves[base].get(n - 1)
                    row[f'{name}_retained'] = None if None in on else on[0] / on[1]
                    row[f'{name}_final_gap'] = None if None in fin else fin[0] - fin[1]
                    row[f'{name}_half'] = M.half_recovery_alignment(
                        curves, sub, arch_key, pert, timing,
                        min_drop=drops[name], key='rho' if name == 'rsa' else 'cka')
                rows.append(row)
    return rows


# ---------------------------------------------------------------- layout

# Partial mode reserves a line under the A title and under the B-D titles
# for what is still missing (zero height otherwise).
A_NOTES, BD_NOTES = (M.NOTE_LINE, M.NOTE_LINE) if PARTIAL else (0.0, 0.0)
W, H = S.TEXT_WIDTH, 3.72 + A_NOTES + BD_NOTES
A_X, A_W = [0.50, 1.72, 2.94, 4.44], 0.90
A_HEAD, A_TOP, A_H = 0.55, 0.80 + A_NOTES, 0.78
B_TOP, B_H = 2.38 + A_NOTES + BD_NOTES, 0.98
BD_HEAD = B_TOP - BD_NOTES - 0.24
MS = 3.4


def _axes(fig, left, top, width, height):
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def _point(ax, x, y, row, **kw):
    ax.plot(x, y, ls='none', zorder=4,
            **S.marker_kw(row['pert'], S.ONSET_COLORS[row['timing']], ms=MS, mew=0.8), **kw)


def _r_note(ax, x, y, **kw):
    r = stats.pearsonr(x, y)[0]
    ax.text(0.04, 0.96, f'r = {r:.2f}', transform=ax.transAxes, ha='left', va='top',
            fontsize=S.FS_NOTE, color=S.NOTE_COLOR, **kw)
    return r


def panel_a(fig, cka):
    for i, (arch_key, arch_label, n_epochs) in enumerate(ARCHES):
        ax = _axes(fig, A_X[i], A_TOP, A_W, A_H)
        for timing in TIMINGS:
            color = S.ONSET_COLORS[timing]
            for pert in PERTS:
                c = cka[F2._run_name(arch_key, pert, timing)]
                for eps in M.stretches(sorted(c)):
                    if len(eps) == 1:
                        ax.plot(eps, [c[e] for e in eps], ls='none', zorder=4, clip_on=False,
                                **S.marker_kw(pert, color, ms=M.MS_SPARSE, mew=0.6))
                    else:
                        ax.plot(eps, [c[e] for e in eps], color=color,
                                lw=F2.LW_PERT, ls=S.pert_linestyle(pert), zorder=3)
        c = cka[F2._baseline_name(arch_key)]
        for eps in M.stretches(sorted(e for e in c if e < n_epochs)):
            if len(eps) == 1:
                ax.plot(eps, [c[e] for e in eps], ls='none', marker='o', ms=M.MS_SPARSE,
                        mew=0, color=S.BASELINE_COLOR, zorder=5, clip_on=False)
            else:
                ax.plot(eps, [c[e] for e in eps], color=S.BASELINE_COLOR,
                        lw=S.LW_BASELINE, zorder=5)
        ax.set_xlim(-1.5 if n_epochs == 90 else -0.6, n_epochs)
        ax.set_xticks([0, 30, 60, 90] if n_epochs == 90 else [0, 10, 20, 30])
        F2.fit_lim(ax, 'y', (-0.02, 0.62), f'A {arch_label}')
        ax.set_yticks([0, 0.3, 0.6])
        ax.set_title(arch_label, fontsize=S.FS_LABEL, pad=2)
        ax.set_xlabel('Training epoch', labelpad=1)
        if i == 0:
            ax.set_ylabel('Behavioral CKA')
    F2._header(fig, 0.02, A_HEAD, 'A', 'Behavioral alignment by training epoch, measured with CKA')
    if PARTIAL:
        M.partial_note(fig, M.NOTE_X, A_HEAD + M.NOTE_LINE,
                       [M.behavioral_coverage(cka, 'CKA')], 1)


def _both_scored(row, quantity):
    """`quantity` is scored under both metrics; in partial mode not every run's is."""
    return row[f'rsa_{quantity}'] is not None and row[f'cka_{quantity}'] is not None


def panel_b(fig, rows):
    ax = _axes(fig, 0.50, B_TOP, 1.00, B_H)
    rows = [r for r in rows if _both_scored(r, 'retained')]
    for row in rows:
        _point(ax, row['rsa_retained'], row['cka_retained'], row)
    F2.fit_lim(ax, 'x', (-0.05, 1.6), 'B')
    F2.fit_lim(ax, 'y', (-0.05, 1.6), 'B')
    top = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, top], [0, top], color=S.REF_COLOR, lw=S.LW_REF, zorder=1)
    ax.set_xticks([0, 0.5, 1, 1.5])
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.set_xlabel('RSA', labelpad=1)
    ax.set_ylabel('CKA')
    _r_note(ax, [r['rsa_retained'] for r in rows], [r['cka_retained'] for r in rows])
    F2._header(fig, 0.02, BD_HEAD, 'B', 'Retained at onset')


def panel_c(fig, rows):
    rows = [r for r in rows if _both_scored(r, 'final_gap')]
    groups = [(r for r in rows if r['arch'] != CLIP_PREFIX), (r for r in rows if r['arch'] == CLIP_PREFIX)]
    for k, (left, width, label) in enumerate([(2.00, 0.80, 'Supervised'), (3.03, 0.52, 'CLIP')]):
        ax = _axes(fig, left, B_TOP, width, B_H)
        sel = list(groups[k])
        for row in sel:
            _point(ax, row['rsa_final_gap'], row['cka_final_gap'], row)
        ax.axhline(0, color=S.REF_COLOR, lw=S.LW_REF, zorder=1)
        ax.axvline(0, color=S.REF_COLOR, lw=S.LW_REF, zorder=1)
        if k == 0:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        else:
            ax.set_xticks([-0.15, 0])
            ax.set_yticks([-0.2, -0.1, 0])
        fmt = mticker.FuncFormatter(lambda v, _: f'{v:g}'.replace('-', F2.MINUS))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_title(label, fontsize=S.FS_LABEL, pad=2)
        ax.set_xlabel('RSA gap', labelpad=1)
        if k == 0:
            ax.set_ylabel('CKA gap')
        _r_note(ax, [r['rsa_final_gap'] for r in sel], [r['cka_final_gap'] for r in sel])
    F2._header(fig, 1.66, BD_HEAD, 'C', 'Gap at the final epoch')


def _timed(half):
    """A half-recovery time exists: scored (partial mode) and with a drop."""
    return half is not None and half[0] is not None


def panel_d(fig, rows):
    ax = _axes(fig, 4.30, B_TOP, 1.10, B_H)
    both = [r for r in rows if _timed(r['rsa_half']) and _timed(r['cka_half'])]
    for row in both:
        _point(ax, row['rsa_half'][0], row['cka_half'][0], row)
    ax.plot([1e-3, 40], [1e-3, 40], color=S.REF_COLOR, lw=S.LW_REF, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.5e-3, 40)
    ax.set_ylim(1.5e-3, 40)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator([0.01, 1]))
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('RSA (epochs)', labelpad=1)
    ax.set_ylabel('CKA (epochs)', labelpad=1)
    rho = stats.spearmanr([r['rsa_half'][0] for r in both], [r['cka_half'][0] for r in both])[0]
    # Undefined while every point shares one value (partial mode: CLIP not timed yet).
    ax.text(0.04, 0.96, 'Spearman n/a' if np.isnan(rho) else f'Spearman {rho:.2f}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=S.FS_NOTE, color=S.NOTE_COLOR)
    if F2.EPOCH_RESOLUTION:
        ax.text(0.96, 0.04, 'seed 43: supervised\ntimed at whole epochs', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=S.FS_NOTE, color=S.NOTE_COLOR)
    F2._header(fig, 3.82, BD_HEAD, 'D', 'Epochs to half-recovery')
    return len(both)


def partial_notes_b_d(fig, rows, cka):
    """B-D share a title line, so their notes share the line beneath it,
    each led by its panel letter."""
    def by_run(quantity):
        return {(r['arch'], r['pert'], r['timing']): r if _both_scored(r, quantity) else None
                for r in rows}
    d_parts = [M.missing_runs(by_run('half'), 'not yet computable')]
    used = complete_baselines(cka)
    if len(used) < len(ARCHES):
        d_parts.append(f'CKA minimum drop from {len(used)} of {len(ARCHES)} baselines')
    parts = [M.panel_part('B', M.missing_runs(by_run('retained'), 'not yet computed')),
             M.panel_part('C', M.missing_runs(by_run('final_gap'), 'not yet computed')),
             M.panel_part('D', ', '.join(p for p in d_parts if p))]
    M.partial_note(fig, M.NOTE_X, BD_HEAD + M.NOTE_LINE, parts, 1)


def figure_key(fig):
    onset = [Line2D([], [], color=S.BASELINE_COLOR, lw=S.LW_BASELINE, label='baseline')]
    onset += [Line2D([], [], color=S.ONSET_COLORS[t], lw=S.LW_DATA, label=f'{t} onset')
              for t in TIMINGS]
    side = {'image_noise': 'input', 'uniform_images': 'input', 'label_shuffle': 'target'}
    perts = [Line2D([], [], color='#444444', lw=S.LW_DATA, ls=S.pert_linestyle(p),
                    label=f'{S.PERT_LABEL[p]} ({side[p]})', **S.marker_kw(p, '#444444', ms=MS))
             for p in PERTS]
    kw = dict(loc='upper center', frameon=False, fontsize=S.FS_LEGEND, handletextpad=0.4,
              columnspacing=1.2, borderaxespad=0)
    fig.legend(handles=onset, ncol=len(onset), bbox_to_anchor=(0.5, 1 - 0.04 / H),
               handlelength=2.2, **kw)
    fig.legend(handles=perts, ncol=len(perts), bbox_to_anchor=(0.5, 1 - 0.20 / H),
               handlelength=3.0, **kw)


# ---------------------------------------------------------------- report

def structure_check(cka):
    """Fig 3's correlations with the alignment ratio recomputed from CKA; None
    if the structure analysis is not computed yet (partial mode)."""
    if not STRUCTURE_JSON.is_file():
        pending(f'{STRUCTURE_JSON.name} (Fig 3 structure correlations)')
        return None
    with STRUCTURE_JSON.open() as f:
        rows = [r for r in json.load(f) if r['pert'] != 'baseline']
    ratio = []
    for r in rows:
        onset = ONSET_EPOCH[r['arch_key']][r['timing']]
        rn = F2._run_name(r['arch_key'], r['pert'], r['timing'])
        ratio.append(cka[rn][onset] / cka[F2._baseline_name(r['arch_key'])][onset])
    ratio = np.array(ratio)
    S_ = np.array([r['category_silhouette_pert'] for r in rows])
    F_ = np.array([r['category_rdm_rho'] for r in rows])
    knn = np.array([r['knn_purity_pert'] / r['knn_purity_base'] for r in rows])
    rsa_ratio = np.array([r['align_retained'] for r in rows])
    out = {'n': len(rows)}
    for name, y in (('RSA', rsa_ratio), ('CKA', ratio)):
        out[name] = {'S': stats.pearsonr(S_, y)[0], 'S|F': F2_partial(y, S_, F_),
                     'S|knn': F2_partial(y, S_, knn), 'F': stats.pearsonr(F_, y)[0],
                     'F|S': F2_partial(y, F_, S_), 'F|knn': F2_partial(y, F_, knn)}
    return out


def F2_partial(y, x, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(rx, ry)[0]


def report(rows, drops, n_d, structure, top1_half):
    sup = [r for r in rows if r['arch'] != CLIP_PREFIX]
    clip = [r for r in rows if r['arch'] == CLIP_PREFIX]
    print(f"minimum drop timed in D: RSA {drops['rsa']:.4f}, CKA {drops['cka']:.4f}")
    if drops['cka_baselines'] < len(ARCHES):
        print(f"  CKA minimum (and the RSA rule) from the {drops['cka_baselines']} of "
              f'{len(ARCHES)} baselines scored at every epoch so far')
    if F2.EPOCH_RESOLUTION:
        # The RSA minimum is Fig 2's constant, set from seed 42's baselines;
        # this is what the same rule gives on this seed's.
        print(f"  RSA rule on this seed's baselines: {drops['rsa_rule']:.4f}")
    for name in ('rsa', 'cka'):
        scored = [r for r in rows if r[f'{name}_retained'] is not None]
        ret = np.array([r[f'{name}_retained'] for r in scored])
        gains = [(r['arch'][:8], r['pert'], r['timing'], round(r[f'{name}_retained'], 2))
                 for r in scored if r[f'{name}_retained'] > 1]
        print(f'{name.upper()}: retained at onset {ret.min():.2f}-{ret.max():.2f}; above baseline: {gains}')
        sg = [r[f'{name}_final_gap'] for r in sup if r[f'{name}_final_gap'] is not None]
        print(f'  supervised final gap {min(sg):+.4f} to {max(sg):+.4f}; CLIP final gaps '
              + ', '.join(f"{r['timing'][0]}/{r['pert'][:5]} {F2.cell(r[f'{name}_final_gap'], '+.3f')}"
                          for r in clip))
        timed = [r for r in sup if _timed(r[f'{name}_half'])]
        exact = [r for r in timed if not r[f'{name}_half'][1]]
        hv = [r[f'{name}_half'][0] for r in exact]
        # At whole-epoch resolution every regained time is the same "within 1".
        span = ('within 1 epoch' if F2.EPOCH_RESOLUTION else
                f'in {min(hv):.4f}-{max(hv):.3f} epochs (median {np.median(hv):.4f})')
        pending_n = sum(r[f'{name}_half'] is None for r in sup)
        print(f'  supervised: {len(timed)} with a drop, {len(exact)} regained half {span}; lower bounds: '
              f"{[(r['arch'][:8], r['pert'], r['timing']) for r in timed if r[f'{name}_half'][1]]}"
              + (f'; not yet computable: {pending_n}' if pending_n else ''))
        if F2.EPOCH_RESOLUTION:
            sooner = 'sooner than top-1: not resolvable at whole epochs'
        else:
            tops = []
            for r in exact:
                t, bound = top1_half[(r['arch'], r['pert'], r['timing'])]
                if not bound:
                    tops.append(r[f'{name}_half'][0] < t)
            sooner = f'sooner than top-1 in {sum(tops)} of {len(tops)}'
        print(f'  {sooner}; CLIP half-recovery '
              + ', '.join(f"{r['timing'][0]}/{r['pert'][:5]} "
                          f"{'n/a' if r[f'{name}_half'] is None else r[f'{name}_half'][0]}"
                          for r in clip))
    print(f'D: {n_d} runs timed under both metrics')
    if structure is None:
        print('Fig 3 structure: n/a (not yet computed)')
        return
    print(f"Fig 3 structure (n={structure['n']}): " + '; '.join(
        f"{m}: r(S)={v['S']:.2f}, partial|F={v['S|F']:.2f}, partial|retained kNN={v['S|knn']:.2f}, "
        f"r(F)={v['F']:.2f}, F partial|S={v['F|S']:.2f}, F partial|retained kNN={v['F|knn']:.2f}"
        for m, v in ((k, structure[k]) for k in ('RSA', 'CKA'))))


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    curves, onset10k = F2.load_epoch_curves(), F2.load_onset_10k()
    sub_top1 = None if F2.EPOCH_RESOLUTION else F2.load_subepoch_data()
    top1_half = {(a, p, t): F2._half_recovery(curves, onset10k, sub_top1, a, p, t)
                 for a in SUPERVISED for t in TIMINGS for p in PERTS}
    rsa = M.load_behavioral_curves()
    cka = load_cka_curves(rsa)
    sub = None if F2.EPOCH_RESOLUTION else M.load_subepoch_alignment(sub_top1)
    drops = {'rsa': M.MIN_ALIGNMENT_DROP, 'cka': median_epoch_change(cka),
             'cka_baselines': len(complete_baselines(cka))}
    if F2.EPOCH_RESOLUTION:
        drops['rsa_rule'] = median_epoch_change(rsa)
    rows = per_run(rsa, cka, sub, drops)

    fig = plt.figure(figsize=(W, H))
    panel_a(fig, cka)
    panel_b(fig, rows)
    panel_c(fig, rows)
    n_d = panel_d(fig, rows)
    figure_key(fig)
    if PARTIAL:
        partial_notes_b_d(fig, rows, cka)
    S.save(fig, OUTDIR, STEM)
    plt.close(fig)
    report(rows, drops, n_d, structure_check(cka), top1_half)


if __name__ == '__main__':
    main()
