"""ImageNet top-1 data, run naming and layout helpers shared by Fig 2 and the supplementary figures.

Loads the ImageNet top-1 around each one-epoch perturbation under one
measurement protocol per quantity:
* epoch curves: the logged 50,000-image validation curve (CLIP: ImageNet linear
  probe on the 768-d CLS features);
* onset drops and sub-epoch recovery: the fixed 10,000-image validation subset
  the sub-epoch checkpoints were scored on (ResNet-50 after BatchNorm
  recalibration);
* final-epoch gaps: the logged curves for both runs (ResNet-50: BatchNorm
  recalibrated on all 50,000 val images).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as S
from arch_runs import (CLIP_PREFIX, PARTIAL, RESULTS, SEED_SUFFIX, baseline_run,
                       clip_imagenet_probe, clip_imagenet_probe_epochs,
                       imagenet_val_curves_path, pending, seeded, widen_for_seed)

S.apply()

IMAGENET_CURVES = imagenet_val_curves_path()
LR_SCHEDULE = RESULTS / 'lr_schedule.json'
BN_RECALIBRATED_DIR = RESULTS / 'bn_recalibrated_val'
# ResNet-50 final-epoch top-1 after BatchNorm recalibration, all 50,000 val images.
BN_RECALIBRATED_FULL_DIR = RESULTS / 'bn_recalibrated_val_50k'
ONSET_10K_DIR = RESULTS / 'onset_val_10k'
SUBEPOCH_DIR = RESULTS / 'subepoch_imagenet_val'

ARCHES = [
    ('resnet50_imagenet',   'RN50',           90),
    ('vitb16_imagenet',     'ViT-B/16',       90),
    ('convnext_b_imagenet', 'ConvNeXt-B',     90),
    (CLIP_PREFIX,           'CLIP ViT-B/32',  32),
]
SUPERVISED = [a[0] for a in ARCHES if a[0] != CLIP_PREFIX]

ONSET_EPOCH = {
    'resnet50_imagenet':   {'early': 5,  'mid': 30, 'late': 70},
    'vitb16_imagenet':     {'early': 5,  'mid': 30, 'late': 70},
    'convnext_b_imagenet': {'early': 5,  'mid': 30, 'late': 70},
    CLIP_PREFIX:           {'early': 5,  'mid': 13, 'late': 28},
}

PERTS = ['image_noise', 'uniform_images', 'label_shuffle']
TIMINGS = ['early', 'mid', 'late']

# Seed 43 has no sub-epoch checkpoints. Its B and supervised D fall back to the
# whole-epoch checkpoints at the onset and one epoch later, both on the same
# 10,000-image subset, and D times them with CLIP's first-evaluated-epoch rule.
EPOCH_RESOLUTION = bool(SEED_SUFFIX)

# Optimizer steps per epoch: 1,281,167 train images, drop_last, global batch
# 1,024 for ResNet-50 and 1,536 for the other two (verified against the
# checkpoints' global_step). Step counts, not the 0-based batch indices the
# sub-epoch checkpoint files are named by.
STEPS_PER_EPOCH = {
    'resnet50_imagenet': 1251,
    'vitb16_imagenet': 834,
    'convnext_b_imagenet': 834,
}
SUBSET_PER_CLASS = 10

BN_ARCHES = {'resnet50_imagenet'}


CASES = {
    ('vitb16_imagenet', 'image_noise', 'late'): 'T',
    ('convnext_b_imagenet', 'label_shuffle', 'late'): 'K',
    (CLIP_PREFIX, 'image_noise', 'early'): 'L',
}

MINUS = '−'
LW_PERT = 0.9
# B's reference levels are wide and pale so they cannot be read as a
# dashed (target-side) or solid (input-side) data line.
REF_LW, REF_ALPHA = 3.2, 0.28
MS_SMALL = 3.4
# ResNet-50's recalibrated baseline top-1 at an onset: a short black bar.


def _run_name(arch_key, pert, timing):
    return seeded(f'{arch_key}_{pert}_{timing}')


def _baseline_name(arch_key):
    return baseline_run(arch_key)


def _subepoch_run_name(arch_key, pert, timing):
    return f'{_run_name(arch_key, pert, timing)}_recovery_dynamics'


def _clip_probe_epochs():
    """{run: epochs} of the CLIP probe the figures read: each onset (A's
    marker, D's drop), the epoch after it (D's first crossing) and the final
    epoch (C). Probed epochs beyond these are drawn and used too."""
    final = dict((a, n) for a, _, n in ARCHES)[CLIP_PREFIX] - 1
    base = _baseline_name(CLIP_PREFIX)
    out = {base: {final}}
    for timing in TIMINGS:
        onset = ONSET_EPOCH[CLIP_PREFIX][timing]
        out[base] |= {onset, onset + 1}
        for pert in PERTS:
            out[_run_name(CLIP_PREFIX, pert, timing)] = {onset, onset + 1, final}
    return out


CLIP_PROBE_EPOCHS = _clip_probe_epochs()


def epoch_list(epochs):
    """'epoch 6' or 'epochs 6-88, 90': how a pending input names its epochs."""
    epochs = sorted(epochs)
    spans, start = [], epochs[0]
    for prev, e in zip(epochs, epochs[1:] + [None]):
        if e != prev + 1:
            spans.append(str(start) if start == prev else f'{start}-{prev}')
            start = e
    return ('epoch ' if len(epochs) == 1 else 'epochs ') + ', '.join(spans)


def cell(v, spec):
    """`v` formatted with `spec`, or 'n/a' at the same width for a value that
    is not computed yet (partial mode), so the printed tables stay aligned."""
    if v is None:
        return 'n/a'.rjust(int(spec.lstrip('+').split('.')[0] or 0))
    return format(v, spec)


def half_label(half):
    """A `_half_recovery` result as the summary tables print it."""
    if half is None:
        return 'n/a'
    t, is_bound = half
    return 'none' if t is None else f'{">" if is_bound else ""}{t:.4g}'


def _read_json(path):
    with path.open() as f:
        return json.load(f)


def _runs_of(arch_key):
    return [_baseline_name(arch_key)] + [
        _run_name(arch_key, p, t) for t in TIMINGS for p in PERTS]


def _load_clip_probe(run_name):
    """{epoch: top-1} of the ImageNet linear probe on CLIP's 768-d CLS features, every
    epoch from the current probe recipe (an older-recipe record raises)."""
    return {ep: clip_imagenet_probe(run_name, ep)['imagenet_val']['top1_accuracy']
            for ep in clip_imagenet_probe_epochs(run_name)}


def load_epoch_curves():
    """{run: {epoch: top-1 fraction}}: the logged 50,000-image validation
    curve for supervised runs, the ImageNet linear probe for CLIP. Nothing
    from another protocol is spliced in."""
    raw = _read_json(IMAGENET_CURVES)['runs']
    curves = {}
    for arch_key, _, _ in ARCHES:
        for rn in _runs_of(arch_key):
            if arch_key == CLIP_PREFIX:
                curves[rn] = _load_clip_probe(rn)
                missing = CLIP_PROBE_EPOCHS[rn] - set(curves[rn])
                if missing:
                    pending(f'{rn} ImageNet probe top-1 {epoch_list(missing)}')
            else:
                curves[rn] = {int(e): v for e, v in raw[rn]['val_acc1'].items()}
    return curves


def load_onset_10k():
    """{(run, epoch): top-1 fraction} at each onset epoch on the fixed
    10,000-image validation subset the sub-epoch checkpoints were scored on;
    under EPOCH_RESOLUTION also each perturbed run one epoch later, the end of
    the span the sub-epoch checkpoints cover. ResNet-50 is after BatchNorm
    recalibration; ViT-B/16 and ConvNeXt-B have no BatchNorm, so their
    recalibrated and as-is values are the same number."""
    out, missing = {}, []
    for arch_key in SUPERVISED:
        root = BN_RECALIBRATED_DIR if arch_key in BN_ARCHES else ONSET_10K_DIR
        for timing in TIMINGS:
            onset = ONSET_EPOCH[arch_key][timing]
            after = [onset, onset + 1] if EPOCH_RESOLUTION else [onset]
            keys = [(_baseline_name(arch_key), onset)] + [
                (_run_name(arch_key, p, timing), e) for p in PERTS for e in after]
            for rn, epoch in keys:
                path = root / f'{rn}_epoch{epoch}.json'
                if not path.is_file():
                    missing.append(str(path))
                    continue
                rec = _read_json(path)
                if rec['val_per_class'] != SUBSET_PER_CLASS:
                    raise ValueError(f'{path}: {rec["val_per_class"]} per class, '
                                     f'expected {SUBSET_PER_CLASS}')
                out[(rn, epoch)] = rec['bn_recalibrated']['val_acc1']
    if missing:
        raise FileNotFoundError(f'{len(missing)} onset files missing, e.g. '
                                f'{missing[:3]}')
    return out


def load_lr_schedules():
    """Return {arch_key: {epoch_int: lr_float}} from each seed-42 baseline run,
    whatever the seed: only seed 42's schedule is extracted, and the seeds'
    runs share every learning-rate setting (hparams.yaml)."""
    raw = _read_json(LR_SCHEDULE)['runs']
    return {arch_key: {int(e): v for e, v in
                       raw[_baseline_name(arch_key).removesuffix(SEED_SUFFIX)].items()}
            for arch_key, _, _ in ARCHES}


def load_subepoch_data():
    """{recovery run: [{step, steps_done, val_acc1, ...}, ...]} for the
    early / mid / late onsets. `step` is the 0-based batch index a checkpoint was saved
    after, so `steps_done` = step + 1 is the optimizer-step count."""
    out = {}
    for arch_key in SUPERVISED:
        for timing in TIMINGS:
            for pert in PERTS:
                rn = _subepoch_run_name(arch_key, pert, timing)
                rec = _read_json(SUBEPOCH_DIR / f'{rn}.json')
                if rec['val_per_class'] != SUBSET_PER_CLASS:
                    raise ValueError(f'{rn}: {rec["val_per_class"]} per class, '
                                     f'expected {SUBSET_PER_CLASS}')
                steps = [dict(s, steps_done=s['step'] + 1) for s in rec['steps']]
                if steps[-1]['steps_done'] != STEPS_PER_EPOCH[arch_key]:
                    raise ValueError(f'{rn}: last checkpoint after '
                                     f'{steps[-1]["steps_done"]} steps, expected '
                                     f'{STEPS_PER_EPOCH[arch_key]} (end of epoch)')
                out[rn] = steps
    return out


def _onset_pair_a(curves, onset10k, arch_key, pert, timing):
    """(baseline, perturbed) top-1 at the onset epoch as panel A draws it:
    ResNet-50 from the recalibrated 10,000-image markers, the rest from their
    lines."""
    onset = ONSET_EPOCH[arch_key][timing]
    base, rn = _baseline_name(arch_key), _run_name(arch_key, pert, timing)
    if arch_key in BN_ARCHES:
        return onset10k[(base, onset)], onset10k[(rn, onset)]
    return curves[base][onset], curves[rn][onset]


def _onset_pair_d(curves, onset10k, arch_key, pert, timing):
    """(baseline, perturbed) top-1 at the onset epoch in the protocol of the
    crossing that follows: the 10,000-image subset for supervised runs, the
    whole-epoch probe for CLIP. None if a CLIP value is not computed yet
    (partial mode; load_epoch_curves recorded it as pending)."""
    onset = ONSET_EPOCH[arch_key][timing]
    base, rn = _baseline_name(arch_key), _run_name(arch_key, pert, timing)
    if arch_key == CLIP_PREFIX:
        if onset not in curves[base] or onset not in curves[rn]:
            return None
        return curves[base][onset], curves[rn][onset]
    return onset10k[(base, onset)], onset10k[(rn, onset)]


def _final_top1(curves, arch_key, run, final):
    """Top-1 at the final epoch: ResNet-50 after BatchNorm recalibration on
    all 50,000 val images, otherwise the logged 50,000-image value. None in
    partial mode for a CLIP probe load_epoch_curves recorded as pending."""
    if arch_key in BN_ARCHES:
        path = BN_RECALIBRATED_FULL_DIR / f'{run}_epoch{final}.json'
        if not path.is_file():
            pending(f'{run} recalibrated top-1 (50,000 images) epoch {final}')
            return None
        rec = _read_json(path)
        if rec['val_per_class'] != 50:
            raise ValueError(f'{path}: {rec["val_per_class"]} per class, expected 50')
        return rec['bn_recalibrated']['val_acc1']
    if final not in curves[run]:
        if PARTIAL and arch_key == CLIP_PREFIX:
            return None
        raise ValueError(f'{run}: no logged top-1 at the final epoch {final}')
    return curves[run][final]


def _final_gap_pp(curves, arch_key, pert, timing, n_epochs):
    """(perturbed - baseline) top-1 in pp at the final training epoch, or
    None if either is not computed yet (partial mode)."""
    final = n_epochs - 1
    pert_top1 = _final_top1(curves, arch_key, _run_name(arch_key, pert, timing), final)
    base_top1 = _final_top1(curves, arch_key, _baseline_name(arch_key), final)
    if pert_top1 is None or base_top1 is None:
        return None
    return (pert_top1 - base_top1) * 100


def _half_crossing(target, start, points):
    """First time `points` [(x, top-1), ...] reach `target`, linearly
    interpolated from the previous point; `start` is the onset point."""
    prev_x, prev_acc = start
    for x, acc in points:
        if acc >= target:
            return prev_x + (target - prev_acc) / (acc - prev_acc) * (x - prev_x)
        prev_x, prev_acc = x, acc
    return None


def _first_epoch_at(curve, onset, target, needed=()):
    """Epochs from `onset` to the first later evaluated epoch of `curve` at or
    above `target` (an upper bound on the crossing), and whether no evaluated
    epoch reaches it (the last one is then a lower bound). None if an epoch
    in `needed` that is not computed yet (partial mode) comes before any
    crossing: it could be the crossing."""
    after = sorted(e for e in set(curve) | set(needed) if e > onset)
    for e in after:
        if e not in curve:
            return None
        if curve[e] >= target:
            return e - onset, False
    return after[-1] - onset, True


def _half_recovery(curves, onset10k, subepoch_data, arch_key, pert, timing):
    """Epochs after the perturbed epoch until top-1 regains half the onset
    drop, and whether that is only a lower bound (not reached by the last
    evaluation). Supervised runs stay on the 10,000-image subset and are
    interpolated between sub-epoch checkpoints; CLIP uses the first evaluated
    whole epoch at or above the target. Under EPOCH_RESOLUTION the supervised
    runs take CLIP's rule over their one evaluated epoch after the onset, and
    a run whose top-1 did not drop has no time, (None, False), as in the
    alignment rows. None (not a pair) if a needed value is not computed yet
    (partial mode)."""
    pair = _onset_pair_d(curves, onset10k, arch_key, pert, timing)
    if pair is None:
        return None
    base_on, pert_on = pair
    rn = _run_name(arch_key, pert, timing)
    if base_on <= pert_on:
        if not EPOCH_RESOLUTION:
            raise ValueError(f'{rn}: no onset drop')
        return None, False
    target = (base_on + pert_on) / 2
    onset = ONSET_EPOCH[arch_key][timing]
    if arch_key == CLIP_PREFIX:
        return _first_epoch_at(curves[rn], onset, target, CLIP_PROBE_EPOCHS[rn])
    if EPOCH_RESOLUTION:
        return _first_epoch_at({onset + 1: onset10k[(rn, onset + 1)]}, onset, target)
    spe = STEPS_PER_EPOCH[arch_key]
    points = [(s['steps_done'] / spe, s['val_acc1'])
              for s in subepoch_data[_subepoch_run_name(arch_key, pert, timing)]]
    t = _half_crossing(target, (0.0, pert_on), points)
    if t is None:
        return points[-1][0], True
    return t, False


# Layout helpers: positions are in inches from the top-left corner of the canvas.


def _axes(fig, left, top, width, height):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, 1 - (top + height) / fh,
                         width / fw, height / fh])


def _fig_y(fig, top_in):
    return 1 - top_in / fig.get_figheight()


def fit_lim(ax, axis, lim, name):
    getattr(ax, f'set_{axis}lim')(
        *widen_for_seed(lim, *getattr(ax.dataLim, f'interval{axis}'), f'{name} {axis}'))


def _header(fig, x_in, base_in, letter, title):
    """Bold letter + title on one line; `base_in` is the title baseline."""
    fw = fig.get_figwidth()
    t = fig.text(x_in / fw, _fig_y(fig, base_in), letter,
                 fontsize=S.FS_LETTER, fontweight='bold',
                 ha='left', va='baseline')
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    x_title = bb.x1 / fig.dpi + 0.05
    fig.text(x_title / fw, _fig_y(fig, base_in), title,
             fontsize=S.FS_TITLE, ha='left', va='baseline')


def _tag(ax, letter, point, at, **kw):
    """Case tag at `at` with a thin leader to the data point it names."""
    ax.plot([point[0], at[0]], [point[1], at[1]], color=S.BASELINE_COLOR,
            lw=0.5, zorder=4, clip_on=kw.get('clip_on', True))
    S.case_tag(ax, at[0], at[1], letter, **kw)


def _group_positions():
    """x position of each (arch, onset, perturbation) marker in C and D."""
    pos = {}
    for g, (arch_key, _, _) in enumerate(ARCHES):
        for k, timing in enumerate(TIMINGS):
            for j, pert in enumerate(PERTS):
                pos[(arch_key, timing, pert)] = g * 12.0 + k * 3.4 + j
    return pos


def print_summary(curves, onset10k, subepoch_data):
    """Print the numbers behind B-D, one row per perturbed run."""
    print(f'\n{"Arch":<14} {"Pert":<15} {"Onset":<6} {"Base@on":>8} '
          f'{"Pert@on":>8} {"Drop(pp)":>8} {"Half(ep)":>9} {"Final(pp)":>11}')
    print('-' * 86)
    for arch_key, arch_label, n_epochs in ARCHES:
        for timing in TIMINGS:
            for pert in PERTS:
                pair = _onset_pair_d(curves, onset10k, arch_key, pert, timing)
                at_onset = ([None] * 3 if pair is None else
                            [pair[0] * 100, pair[1] * 100, (pair[0] - pair[1]) * 100])
                half = half_label(_half_recovery(
                    curves, onset10k, subepoch_data, arch_key, pert, timing))
                gap = _final_gap_pp(curves, arch_key, pert, timing, n_epochs)
                print(f'{arch_label:<14} {pert:<15} {timing:<6} '
                      + ' '.join(cell(v, '8.2f') for v in at_onset)
                      + f' {half:>9} {cell(gap, "11.3f")}')
