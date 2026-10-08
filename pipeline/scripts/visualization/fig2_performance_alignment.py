"""Fig 2 -- ImageNet top-1 and human alignment around a one-epoch perturbation.

Every analysis run on ImageNet top-1 (data from
`make_fig2_test_performance_defs_in_caption.py`) is repeated for behavioral
alignment in a row directly beneath it, and brain alignment is added where it
was measured (the onset and final epochs). Definitions are in the paper's caption.

Two figures. The main-text Fig 2 (STEM):
Panel A: top-1 / behavioral alignment by training epoch, baseline LR beneath.
Panel B: (perturbed - baseline) at the final epoch: top-1, behavioral, brain.
The appendix figure on recovery after the perturbed epoch (STEM_APPENDIX),
whose numbers the Results text quotes:
Panel A: both at the sub-epoch checkpoints of the first epoch after the
         perturbed epoch (supervised models only).
Panel B: epochs to regain half the onset drop: top-1, behavioral.
Panel C: (perturbed - baseline) brain alignment at the onset epoch.

One measurement protocol per line; the top-1 protocols are those of
`make_fig2_test_performance_defs_in_caption.py`, except that panel A's
ResNet-50 top-1 is recalibrated on all 50,000 val images at every epoch
instead of logged. Behavioral alignment is the
Spearman rho between the model's and SPoSE's RDMs over all 1,854 THINGS
images, in each architecture's training preprocessing. Every ResNet-50 value
-- top-1 and alignment, per-epoch, sub-epoch and onset -- is after BatchNorm
recalibration. Brain alignment is the mean in-sample FR-RSA
score over the 27 THINGS-fMRI ROIs. Unlike top-1, behavioral alignment can
rise at the perturbed epoch, or fall by less than the baseline moves between
consecutive epochs; such runs have no half-recovery time (MIN_ALIGNMENT_DROP).

Seed 43 (TP_SEED_SUFFIX=_seed43) has no sub-epoch checkpoints, so there the
appendix figure's A and the supervised half-recovery times come from the
whole-epoch checkpoints at the onset and one epoch later
(F2.EPOCH_RESOLUTION), and the figure says so.

Usage:
    python pipeline/scripts/visualization/fig2_performance_alignment.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fig2_test_performance_defs_in_caption as F2
import make_fig7_summary_defs_in_caption as F7
from arch_runs import CLIP_PREFIX, FIGURES, PARTIAL, RESULTS, SEED_SUFFIX, pending
# The modules above set their own rcParams at import; the print-size style
# has to be applied after them.
import paper_style as S
S.apply()

STEM = 'Fig2_performance_alignment_defs_in_caption'
STEM_APPENDIX = 'FigS_recovery_after_onset'

RDM1854 = RESULTS / 'posthoc_rdm1854'
# Sub-epoch behavioral scores, in the same protocol as each family's
# per-epoch curve (F7.behavioral_curve).
SUBEPOCH_RSA_SUFFIX = {
    'resnet50_imagenet': '_bn_recal_in1k_bilinear',
    'vitb16_imagenet': '_in1k_bicubic',
    'convnext_b_imagenet': '_in1k_bicubic',
}
# This checkpoint was truncated on disk, so the top-1 scorer skipped it; the
# behavioral file still holds a record scored on an earlier checkpoint of that name.
CORRUPT_SUBEPOCH_CKPTS = {('convnext_b_imagenet_image_noise_late_recovery_dynamics', 66)}

ARCHES, SUPERVISED, TIMINGS, PERTS = F2.ARCHES, F2.SUPERVISED, F2.TIMINGS, F2.PERTS
ONSET_EPOCH, STEPS_PER_EPOCH, CASES = F2.ONSET_EPOCH, F2.STEPS_PER_EPOCH, F2.CASES
ARCH_LABEL = {a: label for a, label, _ in ARCHES}
# Smallest onset drop in behavioral rho that D times. The baselines' median
# change between consecutive epochs is 0.003-0.010 (ViT-B/16 0.0098), so a
# smaller drop cannot be told apart from ordinary epoch-to-epoch movement.
MIN_ALIGNMENT_DROP = 0.01
N_EPOCHS = {a: n for a, _, n in ARCHES}


def load_behavioral_curves():
    """{run: {epoch: rho}} for every run; each run must be scored
    from its onset (baseline: epoch 0) through the final epoch. In partial
    mode a run holds the epochs scored so far."""
    out = {}
    for arch_key, _, n_epochs in ARCHES:
        runs = [(F2._baseline_name(arch_key), 0)] + [
            (F2._run_name(arch_key, p, t), ONSET_EPOCH[arch_key][t])
            for t in TIMINGS for p in PERTS]
        for rn, first in runs:
            curve = F7.behavioral_curve(arch_key, rn)
            missing = [e for e in range(first, n_epochs) if e not in curve]
            if missing:
                pending(f'{rn} behavioral {F2.epoch_list(missing)}')
            out[rn] = curve
    return out


def load_recalibrated_top1():
    """{run: {epoch: top-1 fraction}} after BatchNorm recalibration on all
    50,000 val images, for every BatchNorm architecture's run from
    its onset (baseline: epoch 0) through the final epoch. Panel A draws these
    instead of the logged curves, whose running statistics a perturbed epoch
    corrupts. In partial mode a run holds the epochs scored so far."""
    out = {}
    for arch_key in F2.BN_ARCHES:
        n_epochs = N_EPOCHS[arch_key]
        runs = [(F2._baseline_name(arch_key), 0)] + [
            (F2._run_name(arch_key, p, t), ONSET_EPOCH[arch_key][t])
            for t in TIMINGS for p in PERTS]
        for rn, first in runs:
            paths = {e: F2.BN_RECALIBRATED_FULL_DIR / f'{rn}_epoch{e}.json'
                     for e in range(first, n_epochs)}
            missing = [e for e, p in paths.items() if not p.is_file()]
            if missing:
                pending(f'{rn} recalibrated top-1 (50,000 images) {F2.epoch_list(missing)}')
            curve = {}
            for e, path in paths.items():
                if e in missing:
                    continue
                rec = F2._read_json(path)
                if rec['val_per_class'] != 50:
                    raise ValueError(f'{path}: {rec["val_per_class"]} per class, expected 50')
                curve[e] = rec['bn_recalibrated']['val_acc1']
            out[rn] = curve
    return out


def load_subepoch_alignment(subepoch_top1):
    """{recovery run: [{step, steps_done, rho, cka}, ...]} at the checkpoints the
    sub-epoch top-1 was scored on (the steps listed in the sub-epoch top-1
    results), so rendering needs no checkpoints."""
    out = {}
    for arch_key in SUPERVISED:
        for timing in TIMINGS:
            for pert in PERTS:
                rn = F2._subepoch_run_name(arch_key, pert, timing)
                path = (RDM1854 / rn / 'test_things_behavioral_full1854_rsa_summary'
                        f'{SUBEPOCH_RSA_SUFFIX[arch_key]}.json')
                rows = sorted((r['step'], r['rsa_results']['behavioral_rdm']['score'],
                               r['cka_results']['behavioral_cka']['score'])
                              for r in F2._read_json(path)['results']
                              if (rn, r['step']) not in CORRUPT_SUBEPOCH_CKPTS)
                want = [s['step'] for s in subepoch_top1[rn]]
                if [s for s, _, _ in rows] != want:
                    raise ValueError(f'{rn}: behavioral steps {[s for s, _, _ in rows]} '
                                     f'!= top-1 steps {want}')
                out[rn] = [dict(step=s, steps_done=s + 1, rho=v, cka=c) for s, v, c in rows]
    return out


def brain_gap(arch_key, pert, timing, epoch):
    """(perturbed - baseline) mean THINGS-fMRI FR-RSA score at `epoch`, or
    None if either is not computed yet (partial mode)."""
    runs = (F2._run_name(arch_key, pert, timing), F2._baseline_name(arch_key))
    scores = [F7.brain_score(arch_key, rn, epoch) for rn in runs]
    for rn, score in zip(runs, scores):
        if score is None:
            pending(f'{rn} brain alignment epoch {epoch}')
    if None in scores:
        return None
    return scores[0] - scores[1]


def final_behavioral_gap(beh, arch_key, pert, timing):
    """None if either final score is not computed yet (partial mode)."""
    final = N_EPOCHS[arch_key] - 1
    pert_curve = beh[F2._run_name(arch_key, pert, timing)]
    base_curve = beh[F2._baseline_name(arch_key)]
    if final not in pert_curve or final not in base_curve:
        return None
    return pert_curve[final] - base_curve[final]


def half_recovery_alignment(beh, sub_beh, arch_key, pert, timing,
                            min_drop=MIN_ALIGNMENT_DROP, key='rho'):
    """As F2._half_recovery, for behavioral alignment: (epochs, is lower
    bound), or (None, False) when alignment fell by less than `min_drop` at
    the onset (or rose). `key` picks the sub-epoch metric ('rho' or 'cka');
    `beh` must hold the same metric per epoch. Under F2.EPOCH_RESOLUTION the
    supervised runs, like top-1, are timed over the one epoch after the onset.
    None (not a pair) if a needed score is not computed yet (partial mode)."""
    onset = ONSET_EPOCH[arch_key][timing]
    rn = F2._run_name(arch_key, pert, timing)
    base = F2._baseline_name(arch_key)
    if onset not in beh[base] or onset not in beh[rn]:
        return None
    base_on, pert_on = beh[base][onset], beh[rn][onset]
    if base_on - pert_on < min_drop:
        return None, False
    target = (base_on + pert_on) / 2
    if arch_key == CLIP_PREFIX:
        return F2._first_epoch_at(beh[rn], onset, target,
                                  range(onset + 1, N_EPOCHS[arch_key]))
    if F2.EPOCH_RESOLUTION:
        if onset + 1 not in beh[rn]:
            return None
        return F2._first_epoch_at({onset + 1: beh[rn][onset + 1]}, onset, target)
    spe = STEPS_PER_EPOCH[arch_key]
    points = [(s['steps_done'] / spe, s[key])
              for s in sub_beh[F2._subepoch_run_name(arch_key, pert, timing)]]
    t = F2._half_crossing(target, (0.0, pert_on), points)
    if t is None:
        return points[-1][0], True
    return t, False


# Inches from the top-left corner of each canvas; both are the paper's
# text width. Main figure: the by-epoch panel A over the final-epoch gaps (B), with
# the key beside B. Appendix figure: the sub-epoch panel A with the key beside
# it, where CLIP has no sub-epoch column, over the half-recovery times (B) and
# the onset brain-alignment gap (C) side by side. Partial mode reserves lines
# under the panel titles for what each panel is still missing (zero height
# otherwise).

NOTE_LINE = 0.1
A_NOTES, GAP_NOTES, SUB_NOTES, HALF_NOTES = (
    (2 * NOTE_LINE, 2 * NOTE_LINE, NOTE_LINE, 2 * NOTE_LINE) if PARTIAL
    else (0.0, 0.0, 0.0, 0.0))
TEXT_RIGHT = S.TEXT_WIDTH - 0.04
# Partial-mode notes start where the panel titles do.
NOTE_X = 0.19
KEY_X = 3.36
CD_H = 0.38          # one row of a gap / half-recovery panel
CD_ROW_STEP = 0.42

# Main figure.
A_X, A_W = [0.55, 1.75, 2.95, 4.45], 0.90
A_HEAD, A1_TOP, A2_TOP, ROW_H = 0.13, 0.31 + A_NOTES, 0.74 + A_NOTES, 0.39
LR_TOP, LR_H = A2_TOP + ROW_H + 0.02, 0.08
GAP_HEAD = 1.52 + A_NOTES
GAP_TOPS = [GAP_HEAD + 0.11 + GAP_NOTES + k * CD_ROW_STEP for k in range(3)]
C_SUP_X, C_SUP_W = 0.62, 1.95
C_CLIP_X, C_CLIP_W = 2.80, 0.45
H_MAIN = GAP_TOPS[2] + CD_H + 0.16

# Appendix figure.
SUB_HEAD = 0.13
B1_TOP, B2_TOP, B_ROW_H = 0.29 + SUB_NOTES, 0.72 + SUB_NOTES, 0.39
B_X, B_W = [0.55, 1.45, 2.35], 0.80
# The sub-epoch panel under F2.EPOCH_RESOLUTION: a linear x axis over its two
# points (0 and 1), with room right of 1 for the case tags.
EPOCH_B_XLIM, EPOCH_B_TAG_X = (-0.12, 1.45), 1.24
HALF_HEAD = 1.52 + SUB_NOTES
HALF_TOPS = [HALF_HEAD + 0.11 + HALF_NOTES + k * CD_ROW_STEP for k in range(2)]
D_X, D_W = 0.62, 2.10
E_X, E_W, E_HEAD_X = 3.36, 2.10, 2.93
H_APPENDIX = HALF_TOPS[1] + CD_H + 0.16
GROUP_LABELS = ['RN50', 'ViT-B/16', 'ConvNeXt-B', 'CLIP']

# Short enough for 0.4-in rows; the panel titles name both readouts in full.
TOP1_LABEL, ALIGN_LABEL = 'Top-1\n(%)', 'Behavioral\n(ρ)'
ROW_LABEL_FS = 7.0
ALIGN_YLIM, ALIGN_TICKS = (-0.03, 0.56), [0, 0.25, 0.5]
TOP1_YLIM, TOP1_TICKS = (-4, 88), [0, 40, 80]


def _axes(fig, left, top, width, height):
    return F2._axes(fig, left, top, width, height)


def _rho(v, _=None):
    return f'{v:g}'.replace('-', F2.MINUS)


# Behavioral runs with no onset drop sit on their own row at the bottom of D.
NO_DROP_Y = 4e-4
# Partial mode: a scored epoch with no scored neighbour, drawn as a marker
# (unclipped: the final epoch sits within a marker radius of the frame).
MS_SPARSE = 2.4
# (supervised, CLIP) y-limits and ticks for each row of C, and for E.
C_LIMITS = [
    (((-0.62, 0.62), [-0.5, 0, 0.5]), ((-52, 6), [-40, 0])),
    (((-0.016, 0.016), [-0.01, 0, 0.01]), ((-0.19, 0.03), [-0.1, 0])),
    (((-0.009, 0.006), [-0.005, 0, 0.005]), ((-0.06, 0.02), [-0.05, 0])),
]
E_LIMITS = ((-0.21, 0.045), [-0.2, -0.1, 0])


def _lr_strip(fig, i, arch_key, ax, lr_schedules):
    ax_lr = _axes(fig, A_X[i], LR_TOP, A_W, LR_H)
    n_epochs = N_EPOCHS[arch_key]
    lr = lr_schedules[arch_key]
    eps = sorted(e for e in lr if e < n_epochs)
    peak = max(lr[e] for e in eps)
    ax_lr.fill_between(eps, [lr[e] / peak for e in eps], color='#cfcdc8', lw=0)
    ax_lr.plot(eps, [lr[e] / peak for e in eps], color='#77756f', lw=0.6)
    ax_lr.set_xlim(ax.get_xlim())
    ax_lr.set_ylim(0, 1.05)
    ax_lr.set_yticks([])
    ax_lr.spines['left'].set_visible(False)
    ax_lr.set_xticks([0, 30, 60, 90] if n_epochs == 90 else [0, 10, 20, 30])
    if i == 0:
        ax_lr.set_ylabel('LR', rotation=0, ha='right', va='center', labelpad=3)


def _outside_band(xs, ys, lo, hi, n_segments=2):
    """Polyline pieces of (xs, ys) with the stretches of its first
    `n_segments` segments whose values lie inside [lo, hi] cut out; later
    segments are kept whole."""
    pieces, cur = [], []
    for i, ((xa, ya), (xb, yb)) in enumerate(zip(zip(xs, ys), zip(xs[1:], ys[1:]))):
        if ya == yb:
            ta, tb = (0.0, 1.0) if lo <= ya <= hi else (1.0, 1.0)
        else:
            t_lo, t_hi = sorted(((lo - ya) / (yb - ya), (hi - ya) / (yb - ya)))
            ta, tb = max(0.0, t_lo), min(1.0, t_hi)
        if i >= n_segments or ta >= tb:
            cur = cur or [(xa, ya)]
            cur.append((xb, yb))
            continue
        point = lambda t: (xa + t * (xb - xa), ya + t * (yb - ya))
        if ta > 0:
            cur = (cur or [(xa, ya)]) + [point(ta)]
        if len(cur) > 1:
            pieces.append(cur)
        cur = [point(tb), (xb, yb)] if tb < 1 else []
    if len(cur) > 1:
        pieces.append(cur)
    return pieces


def stretches(eps):
    """Sorted `eps` as one line, or in partial mode as runs of consecutive
    epochs, so that no line bridges an epoch that is not scored (yet)."""
    if not PARTIAL:
        return [eps]
    out = []
    for e in eps:
        if out and e == out[-1][-1] + 1:
            out[-1].append(e)
        else:
            out.append([e])
    return out


def partial_note(fig, x_in, base_in, parts, n_lines):
    """Partial mode: what a panel is still missing, one part per line on the
    `n_lines` reserved under its title (the last takes any remainder);
    nothing when the panel is complete."""
    parts = [p for p in parts if p]
    rest = parts[n_lines - 1:]
    lines = parts[:n_lines - 1] + (['; '.join(rest)] if rest else [])
    for i, line in enumerate(lines):
        prefix = f'Seed {SEED_SUFFIX.removeprefix("_seed")}, partial: ' if i == 0 else ''
        fig.text(x_in / fig.get_figwidth(), F2._fig_y(fig, base_in + i * NOTE_LINE),
                 prefix + line, fontsize=S.FS_NOTE, color=S.NOTE_COLOR,
                 ha='left', va='baseline')


def missing_runs(vals, what):
    """'<what> for 8 of 36 runs' over {(arch, pert, timing): value or None},
    naming the architecture when every missing value is one architecture's;
    None when nothing is missing."""
    missing = [k for k, v in vals.items() if v is None]
    if not missing:
        return None
    arches = {k[0] for k in missing}
    if len(arches) > 1:
        return f'{what} for {len(missing)} of {len(vals)} runs'
    arch_key = arches.pop()
    n_runs = sum(k[0] == arch_key for k in vals)
    return f'{ARCH_LABEL[arch_key].split()[0]} {what} for {len(missing)} of {n_runs} runs'


def panel_part(letter, part):
    """A note part for a panel with no note line of its own, led by its letter."""
    return part and f'in {letter}, {part}'


def panel_a(fig, curves, beh, lr_schedules):
    """`curves` is top-1 by epoch in one protocol per run: ResNet-50 after
    BatchNorm recalibration (load_recalibrated_top1), the rest as logged."""
    for i, (arch_key, arch_label, n_epochs) in enumerate(ARCHES):
        ax1 = _axes(fig, A_X[i], A1_TOP, A_W, ROW_H)
        ax2 = _axes(fig, A_X[i], A2_TOP, A_W, ROW_H)
        base = F2._baseline_name(arch_key)
        for timing in TIMINGS:
            onset = ONSET_EPOCH[arch_key][timing]
            color = S.ONSET_COLORS[timing]
            for ax, source, scale in ((ax1, curves, 100), (ax2, beh, 1)):
                target = source[F2._run_name(arch_key, 'label_shuffle', timing)]
                climb = (sorted(target[e] for e in (onset, onset + 1))
                         if onset in target and onset + 1 in target else None)
                for pert in PERTS:
                    rn = F2._run_name(arch_key, pert, timing)
                    curve = source[rn]
                    eps = sorted(curve)
                    # In partial mode a run's onset may not be scored yet.
                    if eps and (eps[0] < onset or (eps[0] > onset and not PARTIAL)):
                        raise ValueError(f'{arch_key} {pert} {timing}: curve starts at '
                                         f'{eps[0]}, not the onset {onset}')
                    kw = dict(color=color, lw=F2.LW_PERT, ls=S.pert_linestyle(pert),
                              zorder=3)
                    for stretch in stretches(eps):
                        ys = [curve[e] * scale for e in stretch]
                        if len(stretch) == 1:
                            if stretch[0] != onset:
                                ax.plot(stretch, ys, ls='none', zorder=6, clip_on=False,
                                        **S.marker_kw(pert, color, ms=MS_SPARSE, mew=0.6))
                        elif pert == 'label_shuffle' or climb is None or stretch[0] != onset:
                            ax.plot(stretch, ys, **kw)
                        else:
                            # At this scale an epoch is ~3 px, so an input-side run
                            # climbing back through the same values as the
                            # label-shuffle climb, at the onset or one epoch later,
                            # sits within a line width of it and shows through its
                            # dashes. That stretch is drawn once, dashed; the
                            # appendix figure's A and B resolve the difference.
                            for piece in _outside_band(stretch, ys, climb[0] * scale,
                                                       climb[1] * scale):
                                ax.plot(*zip(*piece), **kw)
                    if onset in curve:
                        ax.plot(onset, curve[onset] * scale, ls='none', zorder=7,
                                **S.marker_kw(pert, color, ms=F2.MS_SMALL, mew=0.8))

        for ax, curve, scale in ((ax1, curves[base], 100), (ax2, beh[base], 1)):
            eps = sorted(e for e in curve if e < n_epochs)
            for stretch in stretches(eps):
                ys = [curve[e] * scale for e in stretch]
                if len(stretch) == 1:
                    ax.plot(stretch, ys, ls='none', marker='o', ms=MS_SPARSE, mew=0,
                            color=S.BASELINE_COLOR, zorder=6, clip_on=False)
                else:
                    ax.plot(stretch, ys, color=S.BASELINE_COLOR, lw=S.LW_BASELINE,
                            zorder=5, solid_capstyle='round')
        for ax in (ax1, ax2):
            ax.set_xlim(-1.5 if n_epochs == 90 else -0.6, n_epochs)
            ax.tick_params(labelbottom=False)
        F2.fit_lim(ax1, 'y', TOP1_YLIM, f'A top-1 {arch_label}')
        ax1.set_yticks(TOP1_TICKS)
        F2.fit_lim(ax2, 'y', ALIGN_YLIM, f'A behavioral {arch_label}')
        ax2.set_yticks(ALIGN_TICKS)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_rho))
        ax1.set_title(arch_label, pad=2)
        if i == 0:
            ax1.set_ylabel(TOP1_LABEL, fontsize=ROW_LABEL_FS)
            ax2.set_ylabel(ALIGN_LABEL, fontsize=ROW_LABEL_FS)
        elif arch_key == CLIP_PREFIX:
            ax1.set_ylabel('Linear-probe\ntop-1 (%)', fontsize=ROW_LABEL_FS)
        _lr_strip(fig, i, arch_key, ax2, lr_schedules)

        for (a, pert, timing), letter in CASES.items():
            if a != arch_key:
                continue
            rn = F2._run_name(a, pert, timing)
            onset = ONSET_EPOCH[a][timing]
            if letter == 'L':
                last = max(curves[rn])
                F2._tag(ax1, letter, (last, curves[rn][last] * 100),
                        (34.0, curves[rn][last] * 100), clip_on=False)
                F2._tag(ax2, letter, (last, beh[rn][last]), (34.0, beh[rn][last]),
                        clip_on=False)
            else:
                F2._tag(ax1, letter, (onset, curves[rn][onset] * 100),
                        (56, 42) if letter == 'T' else (54, 18))
                F2._tag(ax2, letter, (onset, beh[rn][onset]),
                        (50, 0.46) if letter == 'T' else (52, 0.07))

    F2._header(fig, 0.02, A_HEAD, 'A',
               'ImageNet top-1 and behavioral alignment by training epoch')
    if PARTIAL:
        partial_note(fig, NOTE_X, A_HEAD + NOTE_LINE,
                     [_top1_coverage(curves), behavioral_coverage(beh)], 2)


def _top1_coverage(curves):
    """How many of the checkpoints panel A draws top-1 at are scored, for the
    families whose top-1 is scored per checkpoint (partial mode)."""
    parts = []
    for arch_key in sorted(F2.BN_ARCHES) + [CLIP_PREFIX]:
        need = ({rn: F2.CLIP_PROBE_EPOCHS[rn] for rn in F2._runs_of(arch_key)}
                if arch_key == CLIP_PREFIX else
                {rn: range(_first_scored(arch_key, t), N_EPOCHS[arch_key])
                 for rn, t in _runs_with_timing(arch_key)})
        have = sum(len(set(eps) & set(curves[rn])) for rn, eps in need.items())
        total = sum(len(eps) for eps in need.values())
        if have < total:
            parts.append(f'{have} of {total} {ARCH_LABEL[arch_key].split()[0]}')
    return f'top-1 at {" and ".join(parts)} checkpoints' if parts else None


def _runs_with_timing(arch_key):
    """(run, timing) for the baseline (timing None) and each perturbed run."""
    return [(F2._baseline_name(arch_key), None)] + [
        (F2._run_name(arch_key, p, t), t) for t in TIMINGS for p in PERTS]


def _first_scored(arch_key, timing):
    """First epoch a run is scored at: its onset, or 0 for the baseline."""
    return 0 if timing is None else ONSET_EPOCH[arch_key][timing]


def behavioral_coverage(beh, what='behavioral alignment'):
    """How many behavioral curves are incomplete, and how many of those hold
    only the onset and final epochs (partial mode)."""
    incomplete = sparse = 0
    for arch_key, _, n_epochs in ARCHES:
        for rn, timing in _runs_with_timing(arch_key):
            first = _first_scored(arch_key, timing)
            if len(set(beh[rn]) & set(range(first, n_epochs))) < n_epochs - first:
                incomplete += 1
                sparse += set(beh[rn]) == {first, n_epochs - 1}
    if not incomplete:
        return None
    return (f'{what} incomplete for {incomplete} of {len(beh)} runs '
            f'({sparse} scored at the onset and final epochs only)')


def panel_subepoch(fig, onset10k, subepoch_top1, beh, sub_beh):
    for i, arch_key in enumerate(SUPERVISED):
        ax1 = _axes(fig, B_X[i], B1_TOP, B_W, B_ROW_H)
        ax2 = _axes(fig, B_X[i], B2_TOP, B_W, B_ROW_H)
        base = F2._baseline_name(arch_key)
        spe = STEPS_PER_EPOCH[arch_key]
        for timing in TIMINGS:
            onset = ONSET_EPOCH[arch_key][timing]
            color = S.ONSET_COLORS[timing]
            for pert in PERTS:
                rn = F2._run_name(arch_key, pert, timing)
                if F2.EPOCH_RESOLUTION:
                    rows = ((ax1, onset10k[(rn, onset)] * 100,
                             [onset10k[(rn, onset + 1)] * 100]),
                            (ax2, beh[rn][onset],
                             [beh[rn][onset + 1]] if onset + 1 in beh[rn] else []))
                    xs = [0.0, 1.0]
                else:
                    sub = F2._subepoch_run_name(arch_key, pert, timing)
                    rows = (
                        (ax1, onset10k[(rn, onset)] * 100,
                         [s['val_acc1'] * 100 for s in subepoch_top1[sub]]),
                        (ax2, beh[rn][onset], [s['rho'] for s in sub_beh[sub]]),
                    )
                    xs = [0.0] + [s['steps_done'] / spe for s in subepoch_top1[sub]]
                for ax, y0, ys in rows:
                    ax.plot(xs[:1 + len(ys)], [y0] + ys, color=color, lw=F2.LW_PERT,
                            ls=S.pert_linestyle(pert), zorder=4,
                            **S.marker_kw(pert, color, ms=2.4, mew=0.6))
            for ax, ref in ((ax1, onset10k[(base, onset)] * 100), (ax2, beh[base][onset])):
                ax.axhline(ref, color=color, lw=F2.REF_LW, alpha=F2.REF_ALPHA,
                           solid_capstyle='butt', zorder=2)

        for ax in (ax1, ax2):
            if F2.EPOCH_RESOLUTION:
                ax.set_xlim(*EPOCH_B_XLIM)
                ax.set_xticks([0, 1])
            else:
                ax.set_xscale('symlog', linthresh=0.01, linscale=0.7)
                ax.set_xlim(-0.0009, 1.25)
                ax.set_xticks([0, 0.01, 0.1, 1])
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax1.tick_params(labelbottom=False)
        F2.fit_lim(ax1, 'y', TOP1_YLIM, f'sub-epoch top-1 {ARCH_LABEL[arch_key]}')
        ax1.set_yticks(TOP1_TICKS)
        F2.fit_lim(ax2, 'y', ALIGN_YLIM, f'sub-epoch behavioral {ARCH_LABEL[arch_key]}')
        ax2.set_yticks(ALIGN_TICKS)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_rho))
        ax1.set_title(ARCH_LABEL[arch_key], pad=2)
        if i == 0:
            ax1.set_ylabel(TOP1_LABEL, fontsize=ROW_LABEL_FS)
            ax2.set_ylabel(ALIGN_LABEL, fontsize=ROW_LABEL_FS)
        else:
            for ax in (ax1, ax2):
                ax.tick_params(labelleft=False)
        if i == 1:
            ax2.set_xlabel('Epochs since the perturbed epoch ended '
                           + ('(seed 43: no sub-epoch checkpoints)' if F2.EPOCH_RESOLUTION
                              else '(log scale above 0.01)'), labelpad=1)

        for (a, pert, timing), letter in CASES.items():
            if a != arch_key:
                continue
            if F2.EPOCH_RESOLUTION:
                # Beside the end-of-epoch point, kept clear of the axes edges
                # (the panel title sits just above).
                rn, after = F2._run_name(a, pert, timing), ONSET_EPOCH[a][timing] + 1
                for ax, y in ((ax1, onset10k[(rn, after)] * 100), (ax2, beh[rn].get(after))):
                    if y is None:
                        continue
                    lo, hi = ax.get_ylim()
                    inset = 0.18 * (hi - lo)
                    F2._tag(ax, letter, (1.0, y),
                            (EPOCH_B_TAG_X, min(max(y, lo + inset), hi - inset)))
                continue
            sub = F2._subepoch_run_name(a, pert, timing)
            k = 5 if letter == 'T' else 2
            x = subepoch_top1[sub][k]['steps_done'] / spe
            F2._tag(ax1, letter, (x, subepoch_top1[sub][k]['val_acc1'] * 100),
                    (0.25, 62) if letter == 'T' else (0.09, 50))
            F2._tag(ax2, letter, (x, sub_beh[sub][k]['rho']),
                    (0.25, 0.45) if letter == 'T' else (0.09, 0.40))

    F2._header(fig, 0.02, SUB_HEAD, 'A',
               'Top-1 and behavioral alignment in the epoch following perturbation removal'
               if F2.EPOCH_RESOLUTION else
               'Top-1 and behavioral alignment at sub-epoch training steps '
               'following perturbation removal')
    if PARTIAL:
        after = {(a, p, t): beh[F2._run_name(a, p, t)].get(ONSET_EPOCH[a][t] + 1)
                 for a in SUPERVISED for t in TIMINGS for p in PERTS}
        partial_note(fig, NOTE_X, SUB_HEAD + NOTE_LINE, [missing_runs(
            after, 'behavioral alignment one epoch after the onset not yet computed')], 1)


def figure_key(fig, top, height, band_panel=None):
    """Colour / marker key at KEY_X, from `top` down `height` inches; with
    `band_panel`, also the key to that panel's baseline reference bands."""
    ax = _axes(fig, KEY_X, top, TEXT_RIGHT - KEY_X, height)
    ax.axis('off')
    side = {'image_noise': 'input', 'uniform_images': 'input', 'label_shuffle': 'target'}
    handles = [Line2D([], [], color=S.BASELINE_COLOR, lw=S.LW_BASELINE, label='baseline')]
    handles += [Line2D([], [], color=S.ONSET_COLORS[t], lw=S.LW_DATA,
                       label=f'{t} onset') for t in TIMINGS]
    handles += [Line2D([], [], color='#444444', lw=S.LW_DATA, ls=S.pert_linestyle(p),
                       label=f'{S.PERT_LABEL[p]} ({side[p]})',
                       **S.marker_kw(p, '#444444')) for p in PERTS]
    kw = dict(frameon=False, borderaxespad=0, borderpad=0, handletextpad=0.4,
              labelspacing=0.35)
    top = ax.legend(handles=handles, loc='upper left', ncol=2, handlelength=2.4,
                    columnspacing=0.9, **kw)
    if band_panel is None:
        return
    ax.add_artist(top)
    ref = tuple(Line2D([], [], color=S.ONSET_COLORS[t], lw=F2.REF_LW,
                       alpha=F2.REF_ALPHA, solid_capstyle='butt') for t in TIMINGS)
    ax.legend(handles=[ref], labels=[f'{band_panel}: baseline at the end of\nthe onset epoch'],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
              loc='lower left', handlelength=2.4, fontsize=S.FS_NOTE, **kw)


def _group_axis(ax, n_groups, labels):
    ax.set_xlim(-1.0, (n_groups - 1) * 12.0 + 2 * 3.4 + 2 + 1.0)
    ax.set_xticks([g * 12.0 + 3.4 + 1 for g in range(n_groups)])
    ax.set_xticklabels(labels if labels else [])
    if not labels:
        ax.tick_params(axis='x', length=0)


def _points(ax, values, pos, x0=0.0):
    for (arch_key, pert, timing), v in values.items():
        ax.plot(pos[(arch_key, timing, pert)] - x0, v, ls='none', zorder=5,
                **S.marker_kw(pert, S.ONSET_COLORS[timing], ms=S.MS, mew=0.9))


def panel_final_gap(fig, curves, beh):
    """Main figure B: (perturbed - baseline) at the final epoch; one row per
    readout."""
    pos = F2._group_positions()
    clip0 = pos[(CLIP_PREFIX, 'early', PERTS[0])]
    readouts = [
        ('top1', 'Top-1\n(pp)',
         lambda a, p, t: F2._final_gap_pp(curves, a, p, t, N_EPOCHS[a])),
        ('behavioral', 'Behavioral\n(ρ)', lambda a, p, t: final_behavioral_gap(beh, a, p, t)),
        ('brain', 'Brain\n(ρ)', lambda a, p, t: brain_gap(a, p, t, N_EPOCHS[a] - 1)),
    ]
    out = {}
    for r, (name, ylabel, gap) in enumerate(readouts):
        ax_s = _axes(fig, C_SUP_X, GAP_TOPS[r], C_SUP_W, CD_H)
        ax_c = _axes(fig, C_CLIP_X, GAP_TOPS[r], C_CLIP_W, CD_H)
        vals = {(a, p, t): gap(a, p, t) for a, _, _ in ARCHES for t in TIMINGS
                for p in PERTS}
        out[name] = vals
        shown = {k: v for k, v in vals.items() if v is not None}
        _points(ax_s, {k: v for k, v in shown.items() if k[0] != CLIP_PREFIX}, pos)
        _points(ax_c, {k: v for k, v in shown.items() if k[0] == CLIP_PREFIX}, pos, clip0)
        for ax, (ylim, ticks), group in zip((ax_s, ax_c), C_LIMITS[r], ('supervised', 'CLIP')):
            ax.axhline(0, color=S.REF_COLOR, lw=S.LW_REF, zorder=1)
            F2.fit_lim(ax, 'y', ylim, f'final gap {name} {group}')
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_rho))
        last = r == len(readouts) - 1
        _group_axis(ax_s, 3, [a[1] for a in ARCHES[:3]] if last else None)
        ax_c.set_xlim(-0.9, 2 * 3.4 + 2 + 0.9)
        ax_c.set_xticks([3.4 + 1])
        ax_c.set_xticklabels(['CLIP'] if last else [])
        if not last:
            ax_c.tick_params(axis='x', length=0)
        ax_s.set_ylabel(ylabel)
        for (a, p, t), letter in CASES.items():
            v = vals[(a, p, t)]
            if v is None:
                continue
            ax, x = (ax_c, pos[(a, t, p)] - clip0) if a == CLIP_PREFIX else (ax_s, pos[(a, t, p)])
            lo, hi = ax.get_ylim()
            F2._tag(ax, letter, (x, v), (x, v + (hi - lo) * (-0.28 if letter == 'L' else 0.3)))
    F2._header(fig, 0.02, GAP_HEAD, 'B', 'Gap to baseline at the final epoch of training')
    if PARTIAL:
        partial_note(fig, NOTE_X, GAP_HEAD + NOTE_LINE, [
            missing_runs(out['top1'], 'top-1 not yet computed'),
            missing_runs(out['behavioral'], 'behavioral not yet computed'),
            missing_runs(out['brain'], 'brain not yet computed')], 2)
    return out


def _lower_bound_arrow(ax, x, y, color):
    """Upward arrow above a lower-bound marker, long enough to clear it on
    D's compressed log axis."""
    ax.annotate('', xy=(x, y * 16), xytext=(x, y * 2.2), zorder=6,
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color=color,
                                mutation_scale=5, shrinkA=0, shrinkB=0))


def panel_half_recovery(fig, curves, onset10k, subepoch_top1, beh, sub_beh):
    """Appendix figure B: epochs to regain half the onset drop; top-1 above
    behavioral."""
    pos = F2._group_positions()
    rows = [
        ('Top-1', lambda a, p, t: F2._half_recovery(curves, onset10k, subepoch_top1,
                                                     a, p, t)),
        ('Behavioral', lambda a, p, t: half_recovery_alignment(beh, sub_beh, a, p, t)),
    ]
    out = {}
    for r, (ylabel, half) in enumerate(rows):
        ax = _axes(fig, D_X, HALF_TOPS[r], D_W, CD_H)
        out[ylabel] = {}
        for arch_key, _, _ in ARCHES:
            for timing in TIMINGS:
                color = S.ONSET_COLORS[timing]
                for pert in PERTS:
                    x = pos[(arch_key, timing, pert)]
                    res = half(arch_key, pert, timing)
                    out[ylabel][(arch_key, pert, timing)] = res
                    if res is None:
                        continue
                    t, is_bound = res
                    if t is None:
                        ax.plot(x, NO_DROP_Y, ls='none', zorder=5,
                                **S.marker_kw(pert, color, ms=S.MS - 0.6, mew=0.8))
                        continue
                    ax.plot(x, t, ls='none', zorder=5,
                            **S.marker_kw(pert, color, ms=S.MS, mew=0.9))
                    if is_bound:
                        _lower_bound_arrow(ax, x, t, color)
        ax.set_yscale('log')
        ax.set_ylim(2.2e-4, 40)
        ticks, labels = [0.01, 0.1, 1, 10], ['0.01', '', '1', '']
        if any(res is not None and res[0] is None for res in out[ylabel].values()):
            ticks, labels = [NO_DROP_Y] + ticks, ['none'] + labels
            ax.axhline(NO_DROP_Y * 3, color=S.REF_COLOR, lw=0.4, ls=(0, (1, 1.5)),
                       zorder=1)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.axhline(1, color=S.REF_COLOR, lw=S.LW_REF, zorder=1)
        _group_axis(ax, 4, GROUP_LABELS if r == len(rows) - 1 else None)
        ax.set_ylabel(ylabel, fontsize=ROW_LABEL_FS)
        if F2.EPOCH_RESOLUTION and r == 0:
            ax.text(0.01, 0.17, 'seed 43: supervised timed at whole epochs',
                    transform=ax.transAxes, ha='left',
                    va='bottom', fontsize=S.FS_NOTE, color=S.NOTE_COLOR)
    F2._header(fig, 0.02, HALF_HEAD, 'B', 'Epochs to regain half the onset drop (log)')
    return out


def partial_notes_half_brain(fig, d_vals, e_vals):
    """B's note, with C's appended: C has no note line of its own."""
    partial_note(fig, NOTE_X, HALF_HEAD + NOTE_LINE, [
        missing_runs(d_vals['Top-1'], 'top-1 not yet computable'),
        missing_runs(d_vals['Behavioral'], 'behavioral not yet computable'),
        panel_part('C', missing_runs(e_vals, 'brain not yet computed'))], 2)


def panel_onset_brain(fig):
    """Appendix figure C: (perturbed - baseline) brain alignment at the onset
    epoch, as tall as B's two rows."""
    pos = F2._group_positions()
    ax = _axes(fig, E_X, HALF_TOPS[0], E_W, HALF_TOPS[1] + CD_H - HALF_TOPS[0])
    vals = {(a, p, t): brain_gap(a, p, t, ONSET_EPOCH[a][t])
            for a, _, _ in ARCHES for t in TIMINGS for p in PERTS}
    _points(ax, {k: v for k, v in vals.items() if v is not None}, pos)
    ax.axhline(0, color=S.REF_COLOR, lw=S.LW_REF, zorder=1)
    F2.fit_lim(ax, 'y', E_LIMITS[0], 'onset brain gap')
    ax.set_yticks(E_LIMITS[1])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_rho))
    _group_axis(ax, 4, GROUP_LABELS)
    ax.set_ylabel('Brain\n(ρ)')
    F2._header(fig, E_HEAD_X, HALF_HEAD, 'C', 'Brain-alignment gap at the onset epoch')
    return vals


def print_summary(beh, c_vals, d_vals, e_vals):
    print(f'\n{"Arch":<14} {"Pert":<15} {"Onset":<6} {"b@on":>6} {"p@on":>6} '
          f'{"Half(ep)":>9} {"Final(rho)":>10} {"BrainOn":>8} {"BrainFin":>8}')
    for arch_key, arch_label, _ in ARCHES:
        base = F2._baseline_name(arch_key)
        for timing in TIMINGS:
            onset = ONSET_EPOCH[arch_key][timing]
            for pert in PERTS:
                k = (arch_key, pert, timing)
                rn = F2._run_name(*k)
                half = F2.half_label(d_vals['Behavioral'][k])
                print(f'{arch_label:<14} {pert:<15} {timing:<6} '
                      f'{F2.cell(beh[base].get(onset), "6.3f")} '
                      f'{F2.cell(beh[rn].get(onset), "6.3f")} {half:>9} '
                      f'{F2.cell(c_vals["behavioral"][k], "+10.4f")} '
                      f'{F2.cell(e_vals[k], "+8.4f")} {F2.cell(c_vals["brain"][k], "+8.4f")}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    curves = F2.load_epoch_curves()
    onset10k = F2.load_onset_10k()
    lr_schedules = F2.load_lr_schedules()
    beh = load_behavioral_curves()
    if F2.EPOCH_RESOLUTION:
        subepoch_top1 = sub_beh = None
    else:
        subepoch_top1 = F2.load_subepoch_data()
        sub_beh = load_subepoch_alignment(subepoch_top1)

    fig = plt.figure(figsize=(S.TEXT_WIDTH, H_MAIN))
    panel_a(fig, {**curves, **load_recalibrated_top1()}, beh, lr_schedules)
    c_vals = panel_final_gap(fig, curves, beh)
    figure_key(fig, GAP_TOPS[0], GAP_TOPS[2] + CD_H - GAP_TOPS[0])
    S.save(fig, FIGURES, STEM)
    plt.close(fig)

    fig = plt.figure(figsize=(S.TEXT_WIDTH, H_APPENDIX))
    panel_subepoch(fig, onset10k, subepoch_top1, beh, sub_beh)
    figure_key(fig, B1_TOP - 0.02, B2_TOP + B_ROW_H - B1_TOP, band_panel='A')
    d_vals = panel_half_recovery(fig, curves, onset10k, subepoch_top1, beh, sub_beh)
    e_vals = panel_onset_brain(fig)
    if PARTIAL:
        partial_notes_half_brain(fig, d_vals, e_vals)
    S.save(fig, FIGURES, STEM_APPENDIX)
    plt.close(fig)
    F2.print_summary(curves, onset10k, subepoch_top1)
    print_summary(beh, c_vals, d_vals, e_vals)


if __name__ == '__main__':
    main()
