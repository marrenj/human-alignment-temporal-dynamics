"""Fig 4 -- what happens after a one-epoch perturbation, at print size.

Definitions are in the paper's figure caption.

Two runs, drawn at the paper's text width (5.5 in):

* ViT-B/16, image noise at epoch 70 of 90 (case T);
* CLIP ViT-B/32, uniform images at epoch 5 of 32.

A = TriMap filmstrips of the ImageNet val features around the perturbed
epoch; B = accuracy, representational distance to baseline, behavioral and
brain alignment per epoch.

Writes Fig5_recovery_trajectories_defs_in_caption.{png,pdf} to figures_root.
TP_SEED_SUFFIX=_seed43 draws the same two cases from the seed-43 replicate; adding
TP_PARTIAL=1 draws them from the inputs computed so far, with grey notes on what is missing.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as S
from arch_runs import (CLIP_PREFIX, FIGURES, PARTIAL, PENDING, RESULTS, SEED_SUFFIX,
                       baseline_run, frrsa_dir, imagenet_val_curves_path, pending, probe_root,
                       rdm1854_corrected_suffix, require_current_probe, seeded)
from trimap_imagenet import load_imagenet_category_map

S.apply()

STEM = 'Fig5_recovery_trajectories_defs_in_caption'

# `row_label`: rotated architecture label in A and B (`row_label_b` overrides it in B);
# `row_header`: the run, on its own line above the row's TriMaps; a tracked case shows
# only its circled letter (the letters are defined in the Figure 2 caption).
CASE_T = {
    'row_label': 'ViT-B/16',
    'row_header': None,
    'case': 'T',
    'arch_prefix': 'vitb16_imagenet',
    'run': seeded('vitb16_imagenet_image_noise_late'),
    'baseline': baseline_run('vitb16_imagenet'),
    'pert': 'image_noise',
    'timing': 'late',
    'onset': S.ONSET_EPOCH['vitb16_imagenet']['late'],
    'total_epochs': 90,
    'acc_source': 'imagenet',
    'acc_label_short': 'ImageNet',
    'xlim_c': (62, 91),
}
CASE_U = {
    'row_label': 'CLIP\nViT-B/32',
    'row_label_b': 'CLIP ViT-B/32\nearly uniform images',
    'row_header': 'early uniform images',
    'case': None,
    'arch_prefix': CLIP_PREFIX,
    'run': seeded(f'{CLIP_PREFIX}_uniform_images_early'),
    'baseline': baseline_run(CLIP_PREFIX),
    'pert': 'uniform_images',
    'timing': 'early',
    'onset': S.ONSET_EPOCH[CLIP_PREFIX]['early'],
    'total_epochs': 32,
    'acc_source': 'linear_probe',
    'acc_label_short': 'CIFAR-10\nprobe',
    'xlim_c': (-1, 33),
}

IMAGENET_CURVES = imagenet_val_curves_path()

M2M_ROOTS = {
    'vitb16_imagenet': RESULTS / 'model_to_model_rsa_scratch_pp-in1k_bicubic',
    CLIP_PREFIX: RESULTS / 'model_to_model_rsa_scratch_clip_cls_pp-clip',
}

# Corrected-preprocessing tag of each family's THINGS-fMRI FR-RSA files.
THINGS_FRRSA_TAGS = {
    'vitb16_imagenet': 'in1k_bicubic',
    CLIP_PREFIX: 'clip',
}
N_THINGS_ROIS = 27


# Data loaders

def load_imagenet_accuracy(run_name, n_epochs):
    """Logged 50,000-image val top-1 for epochs 0 .. n_epochs-1 (the ViT-B/16
    baseline's log continues past the 90-epoch schedule)."""
    with IMAGENET_CURVES.open() as f:
        data = json.load(f)
    raw = data['runs'][run_name]['val_acc1']
    epochs = sorted(int(e) for e in raw if int(e) < n_epochs)
    return np.array(epochs), np.array([raw[str(e)] for e in epochs])


def load_linear_probe_accuracy(run_name, dataset='vtab/cifar10'):
    """(epochs, top-1) of `dataset`'s linear probe at every probed epoch, current probe
    recipe only."""
    run_dir = probe_root(run_name) / run_name
    files = sorted(glob.glob(str(run_dir / 'linear_probe_eval_epoch*.json')))
    epochs, scores = [], []
    for fp in files:
        m = re.search(r'epoch(\d+)', fp)
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        if dataset in d.get('results', {}):
            epochs.append(int(m.group(1)))
            scores.append(require_current_probe(d['results'][dataset], f'{fp} {dataset}')['score'])
    order = np.argsort(epochs)
    return np.array(epochs)[order], np.array(scores)[order]


def load_behavioral_alignment(run_name, arch_prefix):
    suffix = rdm1854_corrected_suffix(arch_prefix)
    p = (RESULTS / 'posthoc_rdm1854' / run_name /
         f'test_things_behavioral_full1854_rsa_summary{suffix}.json')
    if not p.is_file():
        print(f'  WARNING: missing {p}')
        return np.array([]), np.array([])
    with p.open() as f:
        data = json.load(f)
    results = data['results']
    epochs = [int(r['epoch']) for r in results]
    scores = [r['rsa_results']['behavioral_rdm']['score'] for r in results]
    order = np.argsort(epochs)
    return np.array(epochs)[order], np.array(scores)[order]


def load_model_to_model_rsa(run_name, arch_prefix, stimulus='things_fmri_720'):
    root = M2M_ROOTS[arch_prefix] / run_name
    if not root.is_dir():
        print(f'  WARNING: missing {root}')
        return np.array([]), np.array([])
    files = sorted(glob.glob(str(root / f'model_to_model_rsa_{stimulus}_epoch*.json')))
    epochs, rhos = [], []
    for fp in files:
        m = re.search(r'epoch(\d+)', fp)
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        # A replicate's perturbed run forked from its own seed's baseline.
        if d['label_b'] != baseline_run(arch_prefix):
            raise ValueError(f'{fp}: distance to {d["label_b"]}, not {baseline_run(arch_prefix)}')
        epochs.append(int(m.group(1)))
        rhos.append(d['rsa']['spearman']['rho'])
    order = np.argsort(epochs)
    return np.array(epochs)[order], np.array(rhos)[order]


def load_brain_alignment(run_name, arch_prefix):
    """THINGS-fMRI layer-reweighted RSA: mean in-sample score over the 27 ROIs."""
    tag = THINGS_FRRSA_TAGS[arch_prefix]
    res_dir = frrsa_dir(run_name)
    files = glob.glob(str(res_dir / f'inference_results_things_frrsa_{tag}_epoch*.json'))
    if not files:
        raise FileNotFoundError(f'no THINGS-fMRI FR-RSA ({tag}) results in {res_dir}')
    epochs, means = [], []
    for fp in files:
        with open(fp) as f:
            blocks = json.load(f)
        scores = [b['insample_score'] for b in blocks.values()]
        if len(scores) != N_THINGS_ROIS or any(v is None for v in scores):
            raise ValueError(f'{fp}: expected {N_THINGS_ROIS} ROI scores, got {scores}')
        epochs.append(int(re.search(r'_epoch(\d+)\.json$', fp).group(1)))
        means.append(float(np.mean(scores)))
    order = np.argsort(epochs)
    return np.array(epochs)[order], np.array(means)[order]


def load_accuracy(case, run=None):
    run = run or case['run']
    if case['acc_source'] == 'imagenet':
        return load_imagenet_accuracy(run, case['total_epochs'])
    return load_linear_probe_accuracy(run)


READOUTS = {
    'accuracy': lambda case, run: load_accuracy(case, run=run),
    'distance': lambda case, run: load_model_to_model_rsa(run, case['arch_prefix']),
    'behavioral': lambda case, run: load_behavioral_alignment(run, case['arch_prefix']),
    'brain': lambda case, run: load_brain_alignment(run, case['arch_prefix']),
}
# What a loader raises for a run with none of that readout computed; the other loaders
# return an empty series.
NOT_COMPUTED = {'accuracy': KeyError, 'brain': FileNotFoundError}

PARTIAL_TAG = f'Seed {SEED_SUFFIX.removeprefix("_seed")}, partial'


def _readout_name(case, kind):
    if kind == 'accuracy':
        return 'ImageNet top-1' if case['acc_source'] == 'imagenet' else 'CIFAR-10 probe top-1'
    return {'distance': 'distance to baseline', 'behavioral': 'behavioral alignment',
            'brain': 'brain alignment'}[kind]


def _pending(what):
    """pending(), once per item: several panels read the same value."""
    if what not in PENDING:
        pending(what)


def _grid_step(case, kind, run):
    """Epoch step of a readout's complete evaluation grid: accuracy on ImageNet and
    behavioral alignment every epoch, the baseline's FR-RSA every 5th epoch and the last.
    None where each run has its own sparse set (the CIFAR-10 probe, distance to baseline,
    the perturbed run's FR-RSA): there only the epochs the figure reports are checked."""
    if kind == 'behavioral' or (kind == 'accuracy' and case['acc_source'] == 'imagenet'):
        return 1
    if kind == 'brain' and run == case['baseline']:
        return 5
    return None


def _coverage(case, kind, run, ep):
    """(epochs evaluated, epochs on the complete grid, the grid's missing epochs) within the
    run's plotted window, or None when the grid is complete or unknown."""
    step = _grid_step(case, kind, run)
    if step is None:
        return None
    last = case['total_epochs'] - 1
    lo, hi = ((max(case['xlim_c'][0], 0), min(case['xlim_c'][1], last))
              if run == case['baseline'] else (case['onset'], last))
    have = {int(e) for e in ep if lo <= e <= hi}
    grid = {e for e in range(lo, hi + 1) if e % step == 0} | {hi}
    missing = sorted(grid - have)
    if not missing:
        return None
    return len(have), len(grid | have), missing


def _span(epochs):
    """[71, 72, 73, 80] -> '71-73, 80'."""
    runs = []
    for e in epochs:
        if runs and e == runs[-1][1] + 1:
            runs[-1][1] = e
        else:
            runs.append([e, e])
    return ', '.join(f'{a}-{b}' if b > a else f'{a}' for a, b in runs)


def read(case, kind, run):
    """(epochs, values) of one readout of `run`. Partial mode returns what is computed so far
    and records the rest as pending: the whole readout, or the epochs its grid lacks."""
    try:
        ep, vals = READOUTS[kind](case, run)
    except NOT_COMPUTED.get(kind, ()):
        _pending(f'{run} {_readout_name(case, kind)}: all epochs')
        return np.array([]), np.array([])
    if PARTIAL:
        cov = _coverage(case, kind, run, ep)
        if not len(ep):
            _pending(f'{run} {_readout_name(case, kind)}: all epochs')
        elif cov:
            _pending(f'{run} {_readout_name(case, kind)} epochs {_span(cov[2])}')
    return ep, vals


def _value_pending(case, kind, run, ep_v, epoch):
    """Partial mode: record a value the figure reports but that is not computed yet, unless
    the readout's own record (all epochs, or its grid's missing epochs) already covers it."""
    if len(ep_v[0]) and _grid_step(case, kind, run) is None:
        _pending(f'{run} {_readout_name(case, kind)} epoch {epoch}')


def _pct(ep_v):
    ep, v = ep_v
    return ep, 100.0 * v


def _at(ep, vals, epoch):
    idx = np.where(ep == epoch)[0]
    return float(vals[idx[0]]) if len(idx) else None


def _diff_at(pert, base, epoch, what):
    """Perturbed minus baseline at exactly `epoch`; both must have been evaluated."""
    p, b = _at(*pert, epoch), _at(*base, epoch)
    if p is None or b is None:
        raise ValueError(f'{what}: no {"perturbed" if p is None else "baseline"} '
                         f'value at epoch {epoch}')
    return p - b




# Layout + text helpers (local; paper_style is shared and not edited here)

def _axes_in(fig, left, top, width, height):
    """Axes placed in inches from the figure's left/top edge."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    return fig.add_axes([left / W, 1.0 - (top + height) / H, width / W, height / H])


def _y(fig, top_in):
    return 1.0 - top_in / fig.get_figheight()


def _note(ax, x, y, text, color=S.NOTE_COLOR, ha='left', va='top'):
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=S.FS_TICK,
                   color=color, ha=ha, va=va, linespacing=1.1, zorder=15)


NOTE_CORNERS = [(0.98, 0.97, 'right', 'top'), (0.02, 0.97, 'left', 'top'),
                (0.98, 0.05, 'right', 'bottom'), (0.02, 0.05, 'left', 'bottom')]


def _covers_data(ax, note):
    """Whether `note` lies over any drawn point or line segment of `ax`, with a marker's
    radius of margin."""
    renderer = ax.figure.canvas.get_renderer()
    box = note.get_window_extent(renderer).padded(4 * ax.figure.dpi / 72)
    for line in ax.get_lines():
        xy = line.get_xydata()
        if line.get_linestyle() not in ('None', 'none', '') and len(xy) > 1:
            t = np.linspace(0, 1, 16)[:, None, None]
            xy = (xy[:-1] + t * (xy[1:] - xy[:-1])).reshape(-1, 2)
        if box.count_contains(line.get_transform().transform(xy)):
            return True
    return False


def _clear_of_data(ax, note):
    """The notes' spots were chosen on seed 42's complete data; partial mode moves a note that
    covers the data drawn so far to the first corner where it covers none."""
    if not _covers_data(ax, note):
        return
    for x, y, ha, va in NOTE_CORNERS:
        note.set_position((x, y))
        note.set_ha(ha)
        note.set_va(va)
        if not _covers_data(ax, note):
            return
    print(f'  note {note.get_text()!r} covers data in every corner')


def _signed(v, fmt):
    text = format(abs(v), fmt)
    if float(text) == 0:
        return text
    return ('+' if v > 0 else '−') + text


def _star(ax, x, y, color, zorder=8):
    ax.plot([x], [y], marker='*', ms=S.MS_STAR + 1.0, color=color,
            markeredgecolor='white', markeredgewidth=0.4, ls='none',
            zorder=zorder, clip_on=False)


def _not_yet_computed(ax, y=0.5):
    ax.text(0.5, y, 'not yet computed', transform=ax.transAxes, ha='center',
            va='center', fontsize=S.FS_NOTE, color=S.NOTE_COLOR)


def _segments(ep, gap):
    """Index runs of `ep` with no step wider than `gap` epochs -- so a partly computed
    series never draws a line across epochs not evaluated yet; one run when `gap` is None."""
    if gap is None:
        return [slice(None)]
    return np.split(np.arange(len(ep)), np.flatnonzero(np.diff(ep) > gap) + 1)


def _pert_line(ax, ep, vals, case, color, dense_ms=2.6, gap=None):
    """Perturbed run over training epochs: line + perturbation-type markers
    (markers only when fewer than 5 epochs were evaluated), star at onset.
    With `gap`, lines join only the evaluated epochs of the grid that are adjacent."""
    kw = S.marker_kw(case['pert'], color, ms=dense_ms if len(ep) > 8 else S.MS)
    for seg in _segments(ep, gap):
        s_ep = ep[seg]
        ls = '-' if len(s_ep) >= (5 if gap is None else 2) else 'none'
        marked = [i for i, e in enumerate(s_ep) if e != case['onset']]
        ax.plot(s_ep, vals[seg], ls=ls, color=color, lw=S.LW_DATA, zorder=5,
                markevery=marked, **kw)
    v0 = _at(ep, vals, case['onset'])
    if v0 is not None:
        _star(ax, case['onset'], v0, color)


def _baseline_line(ax, ep, vals, with_markers=False, gap=None):
    kw = (dict(marker='o', ms=2.2, markeredgewidth=0)
          if with_markers or gap is not None else {})
    for seg in _segments(ep, gap):
        ax.plot(ep[seg], vals[seg], '-', color=S.BASELINE_COLOR, lw=S.LW_BASELINE * 0.8,
                zorder=3, **kw)


def _pad_ylim(ax, frac=0.08):
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - frac * (y1 - y0), y1 + frac * (y1 - y0))


def _row_header(fig, ax, case):
    """The run a TriMap row shows, left-aligned above its frame titles, led by the
    circled case letter when the run is one of the tracked cases."""
    bb = ax.get_position()
    x = bb.x0 * fig.get_figwidth()
    y = (1 - bb.y1) * fig.get_figheight() - 0.16
    if case['case']:
        fig.text((x + 0.06) / fig.get_figwidth(), _y(fig, y), case['case'], ha='center',
                 va='baseline', fontsize=S.FS_NOTE + 0.5, fontweight='bold',
                 bbox=dict(boxstyle='circle,pad=0.18', facecolor='white',
                           edgecolor='#111111', linewidth=0.7))
        x += 0.18
    if case['row_header']:
        fig.text(x / fig.get_figwidth(), _y(fig, y), case['row_header'], ha='left',
                 va='baseline', fontsize=S.FS_LABEL)


def _row_label(fig, ax, text, x_in=0.15, fontsize=S.FS_LABEL):
    bb = ax.get_position()
    fig.text(x_in / fig.get_figwidth(), 0.5 * (bb.y0 + bb.y1), text,
             rotation=90, ha='center', va='center', fontsize=fontsize,
             linespacing=1.1)


# A: TriMap filmstrips

TRIMAP_COORDS_DIR = RESULTS / 'trimap'
TRIMAP_SYNSETS_DIR = RESULTS / 'imagenet_embeddings'

# Colours and wording shared with Fig 3's TriMaps: the five largest WordNet
# categories keep a hue, the other ten share one (S.trimap_scatter).
TRIMAP_GROUP_COLORS = S.TRIMAP_CATEGORY_COLORS
OTHER = 'other'

FRAME_ROLES = ['baseline', 'perturbed', 'perturbed', 'perturbed', 'baseline']

_category_cache = {}


def _trimap_frames(case):
    onset, last = case['onset'], case['total_epochs'] - 1
    bl, pt = case['baseline'], case['run']
    return [(bl, onset), (pt, onset), (pt, min(onset + 5, last)), (pt, last), (bl, last)]


def _load_categories(run, epoch):
    key = (run, epoch)
    if key not in _category_cache:
        syn_path = TRIMAP_SYNSETS_DIR / f'{run}_epoch{epoch}_synsets.json'
        with syn_path.open() as f:
            synset_ids = json.load(f)
        categories, _ = load_imagenet_category_map(synset_ids)
        _category_cache[key] = (np.array(categories), np.array(synset_ids))
    return _category_cache[key]


def _group_of(category):
    if category in TRIMAP_GROUP_COLORS:
        return category
    return 'Uncategorized' if category == 'Uncategorized' else OTHER


def trimap_category_stats(run, epoch):
    """Number of categories sharing the 'other' colour, the largest image
    share of any one of them, and the number of images."""
    categories, _ = _load_categories(run, epoch)
    other = {c for c in categories if _group_of(c) == OTHER}
    max_share = max(np.mean(categories == c) for c in other)
    return len(other), max_share, len(categories)


def _frame_caption(case, run, epoch):
    beh = _at(*read(case, 'behavioral', run), epoch)
    if beh is None and not PARTIAL:
        raise ValueError(f'no behavioral alignment for {run} at epoch {epoch}')
    acc = _at(*_pct(read(case, 'accuracy', run)), epoch)
    beh_txt = f'ρ {beh:.2f}' if beh is not None else 'ρ pending'
    # The probe is run at a few epochs by design (the caption says which frame has
    # none); ImageNet top-1 is logged every epoch.
    if acc is not None:
        return f'{beh_txt} · {acc:.1f}%'
    if PARTIAL and _grid_step(case, 'accuracy', run):
        return f'{beh_txt} · pending'
    return beh_txt


def _load_frame(run, epoch):
    try:
        return np.load(TRIMAP_COORDS_DIR / f'trimap_fig5b_{run}_epoch{epoch}_coords.npy')
    except FileNotFoundError:
        pending(f'{run} TriMap epoch {epoch}')
        return None


def _aligned_row(case):
    """The row's frames, each frame's TriMap Procrustes-aligned onto a reference (None for a
    frame not computed yet), and the reference: the row's first frame, or in partial mode
    the first baseline frame computed.

    Every frame embeds the same 50,000 images in the same order (one
    ImageFolder pass, shuffle off), so rows correspond point for point.
    """
    frames = _trimap_frames(case)
    raw = [_load_frame(run, epoch) for run, epoch in frames]
    # The row's limits come from the reference; a collapsed perturbed map would crop the rest.
    order = sorted(range(len(raw)), key=lambda i: frames[i][0] != case['baseline'])
    ref = next((raw[i] for i in order if raw[i] is not None), None)
    return frames, [c if c is None or c is ref else S.procrustes_onto(ref, c) for c in raw], ref


def _shared_limits(ref, box_aspect):
    """Row-wide limits from the first frame's central 99%, equal scale on x and y.

    The collapsed CLIP maps are near 1-D curves with sparse, long tails; limits
    spanning every frame would shrink the baseline maps to a third of the frame.
    """
    lo = np.percentile(ref, 0.5, axis=0)
    hi = np.percentile(ref, 99.5, axis=0)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 1.06
    if half[0] / half[1] < box_aspect:
        half[0] = half[1] * box_aspect
    else:
        half[1] = half[0] / box_aspect
    return mid, half


def plot_filmstrip(axes, case):
    """Draw one row; return the largest fraction of a frame's points out of view, and the
    number of frames not computed yet."""
    frames, coords_list, ref = _aligned_row(case)
    drawn = [c for c in coords_list if c is not None]
    bb = axes[0].get_position()
    fig = axes[0].figure
    if drawn:
        mid, half = _shared_limits(ref, (bb.width * fig.get_figwidth())
                                   / (bb.height * fig.get_figheight()))
    out_of_view = max((np.mean(np.any(np.abs(c - mid) > half, axis=1)) for c in drawn),
                      default=0.0)
    for i, (ax, (run, epoch), coords) in enumerate(zip(axes, frames, coords_list)):
        if coords is None:
            _not_yet_computed(ax)
        else:
            categories, _ = _load_categories(run, epoch)
            S.trimap_scatter(ax, coords, categories, s=0.12)
            ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
            ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
        ax.set_xticks([]); ax.set_yticks([])
        is_baseline = run == case['baseline']
        edge = S.BASELINE_COLOR if is_baseline else S.ONSET_COLORS[case['timing']]
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(edge)
            sp.set_linewidth(1.4 if i == 1 else 0.8)
        ax.text(0.5, -0.05, _frame_caption(case, run, epoch),
                transform=ax.transAxes, ha='center', va='top',
                fontsize=S.FS_NOTE, color='#333333')
        ax.text(0.5, 1.04, f'{FRAME_ROLES[i]}, epoch {epoch}',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=S.FS_NOTE, color='#333333')
    return out_of_view, len(coords_list) - len(drawn)


def _category_handles(n_other, max_share_pct):
    def h(color, label):
        return Line2D([], [], ls='none', marker='o', ms=3.4, markeredgewidth=0,
                      color=color, label=label)
    hs = [h(TRIMAP_GROUP_COLORS[g], g.lower()) for g in TRIMAP_GROUP_COLORS]
    hs.append(h(S.TRIMAP_OTHER_COLOR,
                f'{n_other} smaller categories (< {max_share_pct}% of images each)'))
    hs.append(h(S.TRIMAP_UNCATEGORIZED_COLOR, 'uncategorized'))
    return hs


# B: per-epoch readouts

C_COLUMNS = ['Accuracy (%)', 'Distance to baseline', 'Behavioral alignment (ρ)',
             f'Brain alignment ({N_THINGS_ROIS} ROIs)']


def _gap(case, kind, run, ep):
    """Partial mode draws an incomplete series without lines across its grid's gaps."""
    return _grid_step(case, kind, run) if PARTIAL and _coverage(case, kind, run, ep) else None


def _partial_note(case, kind, bl_ep, pt_ep):
    """What a panel's series lack in partial mode: 'epochs computed: / baseline 4 of 32', or
    'perturbed not yet computed'; empty when both are complete."""
    counts, absent = [], []
    for role, run, ep in (('baseline', case['baseline'], bl_ep), ('perturbed', case['run'], pt_ep)):
        cov = _coverage(case, kind, run, ep)
        if not len(ep):
            absent.append(f'{role} not yet computed')
        elif cov:
            counts.append(f'{role} {cov[0]} of {cov[1]}')
    return '\n'.join((['epochs computed:'] + counts if counts else []) + absent)


def plot_c_row(axes, case):
    """Draw one row; return each panel's partial-mode note on what is not computed yet."""
    run, bl_run = case['run'], case['baseline']
    color = S.ONSET_COLORS[case['timing']]
    x0, x1 = case['xlim_c']

    def _clip(ep, v):
        m = (ep >= x0) & (ep <= x1)
        return ep[m], v[m]

    ax = axes[0]
    bl = _clip(*_pct(read(case, 'accuracy', bl_run)))
    pt = _pct(read(case, 'accuracy', run))
    _baseline_line(ax, *bl, with_markers=len(bl[0]) < 20,
                   gap=_gap(case, 'accuracy', bl_run, bl[0]))
    _pert_line(ax, *pt, case, color, gap=_gap(case, 'accuracy', run, pt[0]))
    ax.set_ylabel(case['acc_label_short'], linespacing=1.1)
    notes = [_partial_note(case, 'accuracy', bl[0], pt[0])]

    ax = axes[1]
    m_ep, m_rho = read(case, 'distance', run)
    ax.axhline(0, color=S.BASELINE_COLOR, lw=S.LW_BASELINE * 0.8, zorder=3)
    _pert_line(ax, m_ep, 1.0 - m_rho, case, color)
    if PARTIAL and not len(m_ep):
        _not_yet_computed(ax, y=0.75)
        ax.tick_params(axis='y', left=False, labelleft=False)
    notes.append('')

    for ax, kind, markers in ((axes[2], 'behavioral', False), (axes[3], 'brain', True)):
        bl = _clip(*read(case, kind, bl_run))
        pt = read(case, kind, run)
        _baseline_line(ax, *bl, with_markers=markers, gap=_gap(case, kind, bl_run, bl[0]))
        _pert_line(ax, *pt, case, color, gap=_gap(case, kind, run, pt[0]))
        notes.append(_partial_note(case, kind, bl[0], pt[0]))

    for a in axes:
        a.set_xlim(x0, x1)
        a.xaxis.set_major_locator(MultipleLocator(10))
        a.yaxis.set_major_locator(MaxNLocator(3))
        _pad_ylim(a)
    return notes


def _missing_at(case, kind, sides, epoch):
    """Partial mode: whether any of the (run, series) `sides` lacks `epoch`, recording it."""
    missing = False
    for run, ep_v in sides:
        if _at(*ep_v, epoch) is None:
            _value_pending(case, kind, run, ep_v, epoch)
            missing = True
    return missing


def c_gap_notes(case, epochs_per_column):
    """Per column: 'E: perturbed - baseline' lines (distance column: its value).
    Partial mode leaves out an epoch either side has not computed yet."""
    run, bl_run = case['run'], case['baseline']
    m_ep_v = read(case, 'distance', run)
    pairs = [
        (_pct(read(case, 'accuracy', run)), _pct(read(case, 'accuracy', bl_run)), '.1f', ' pp',
         'accuracy', 'accuracy'),
        None,
        (read(case, 'behavioral', run), read(case, 'behavioral', bl_run), '.3f', '',
         'behavioral alignment', 'behavioral'),
        (read(case, 'brain', run), read(case, 'brain', bl_run), '.3f', '',
         'brain alignment', 'brain'),
    ]
    notes = []
    for pair, epochs in zip(pairs, epochs_per_column):
        lines = []
        for e in epochs:
            if pair is None:
                if PARTIAL and _missing_at(case, 'distance', [(run, m_ep_v)], e):
                    continue
                rho = _at(*m_ep_v, e)
                if rho is None:
                    raise ValueError(f'distance to baseline: no {run} value at epoch {e}')
                lines.append(f'{e}: {1.0 - rho:.2f}')
            else:
                pert, base, fmt, unit, what, kind = pair
                if PARTIAL and _missing_at(case, kind, [(run, pert), (bl_run, base)], e):
                    continue
                diff = _diff_at(pert, base, e, f'{what} ({run} vs {bl_run})')
                lines.append(f'{e}: {_signed(diff, fmt)}{unit}')
        notes.append('\n'.join(lines))
    return notes


# Figure

# Partial mode opens a strip under A and a band under each row of B for the notes on what
# is not computed yet.
A_STRIP, NOTE_BAND = (0.14, 0.31) if PARTIAL else (0.0, 0.0)
# ROW_HEAD: the line above each row of TriMaps that names its run.
ROW_HEAD = 0.15
FIG_H = 4.12 + 2 * ROW_HEAD + A_STRIP + 2 * NOTE_BAND

# Vertical layout, inches from the top edge.
A_HEAD = 0.50
B_TOP, F_H, F_GAP, B_LEFT, CAP_H = 0.70 + ROW_HEAD, 0.46, 0.10, 0.36, 0.28 + ROW_HEAD
F_W = (S.TEXT_WIDTH - 0.05 - B_LEFT - 4 * F_GAP) / 5
C_TOP, C_H, C_VGAP = 2.56 + 2 * ROW_HEAD + A_STRIP, 0.50, 0.24 + NOTE_BAND
C_LEFT, C_W, C_HGAP = 0.70, 0.89, 0.37


def _key(fig):
    row1 = [Line2D([], [], color=S.BASELINE_COLOR, lw=S.LW_BASELINE,
                   label='unperturbed baseline')]
    for case in (CASE_U, CASE_T):
        row1.append(Line2D([], [], color=S.ONSET_COLORS[case['timing']], lw=S.LW_DATA,
                           label=f'{case["timing"]} onset '
                                 f'(epoch {case["onset"]})'))
    row1.append(Line2D([], [], color='#444444', ls='none', marker='*',
                       ms=S.MS_STAR, markeredgewidth=0,
                       label='end of the perturbed epoch'))
    row2 = [Line2D([], [], color='#444444', ls='none', label=S.PERT_LABEL[p],
                   **S.marker_kw(p, '#444444'))
            for p in (CASE_T['pert'], CASE_U['pert'])]
    for hs, top in ((row1, 0.02), (row2, 0.19)):
        leg = fig.legend(handles=hs, loc='upper center', ncol=len(hs),
                         bbox_to_anchor=(0.5, _y(fig, top)), handlelength=1.8,
                         columnspacing=1.3, handletextpad=0.5, borderaxespad=0)
        fig.add_artist(leg)


def _header(fig, top, letter, title):
    """Bold panel letter + plain title on one baseline, `top` inches down."""
    t = fig.text(0.03 / S.TEXT_WIDTH, _y(fig, top), letter, fontsize=S.FS_LETTER,
                 fontweight='bold', ha='left', va='baseline')
    fig.canvas.draw()
    x1 = t.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi + 3.5 / 72
    fig.text(x1 / S.TEXT_WIDTH, _y(fig, top), title, fontsize=S.FS_TITLE,
             ha='left', va='baseline')


def make_figure():
    fig = plt.figure(figsize=(S.TEXT_WIDTH, FIG_H))
    fig.canvas.draw()
    _key(fig)

    # --- A ---
    strips = []
    for r in range(2):
        top = B_TOP + r * (F_H + CAP_H)
        strips.append([_axes_in(fig, B_LEFT + i * (F_W + F_GAP), top, F_W, F_H)
                       for i in range(5)])
    print('A: TriMap filmstrips')
    (oov_t, missing_t), (oov_u, missing_u) = (plot_filmstrip(strips[0], CASE_T),
                                              plot_filmstrip(strips[1], CASE_U))
    out_of_view = max(oov_t, oov_u)
    print(f'  points out of view (caption): <= {np.ceil(100 * out_of_view):.0f}%')
    for strip, case in ((strips[0], CASE_T), (strips[1], CASE_U)):
        _row_label(fig, strip[0], case['row_label'], fontsize=S.FS_TICK)
        _row_header(fig, strip[0], case)

    n_other, max_share, n_img = trimap_category_stats(*_trimap_frames(CASE_T)[0])
    b_bottom = B_TOP + 2 * F_H + CAP_H + 0.15
    fig.legend(handles=_category_handles(n_other, int(np.ceil(100 * max_share))),
               loc='upper center', ncol=7, bbox_to_anchor=(0.5, _y(fig, b_bottom)),
               fontsize=S.FS_NOTE, handlelength=0.8, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0)

    # --- B ---
    grid = [[_axes_in(fig, C_LEFT + j * (C_W + C_HGAP), C_TOP + r * (C_H + C_VGAP), C_W, C_H)
             for j in range(4)] for r in range(2)]
    for ax, title in zip(grid[0], C_COLUMNS):
        ax.set_title(title, fontsize=S.FS_TICK, pad=3)
    print('B: per-epoch readouts')
    note_spec = {
        'T': ([[70, 71], [70, 71, 89], [70, 71], [70, 89]],
              [(0.98, 0.05, 'right', 'bottom'), (0.98, 0.97, 'right', 'top'),
               (0.98, 0.97, 'right', 'top'), (0.98, 0.97, 'right', 'top')]),
        'U': ([[31]] * 4,
              [(0.98, 0.05, 'right', 'bottom'), (0.98, 0.97, 'right', 'top'),
               (0.98, 0.70, 'right', 'center'), (0.98, 0.05, 'right', 'bottom')]),
    }
    # CLIP's CIFAR-10 probe is drawn at every epoch from its onset on, so its accuracy note
    # goes on two lines to fit the empty lower-right corner beside the curve.
    note_break = {'U': {0}}
    for row, case, key in ((grid[0], CASE_T, 'T'), (grid[1], CASE_U, 'U')):
        partial_notes = plot_c_row(row, case)
        epochs, positions = note_spec[key]
        for j, (ax, text, (x, y, ha, va)) in enumerate(zip(row, c_gap_notes(case, epochs),
                                                           positions)):
            if j in note_break.get(key, ()):
                text = text.replace(': ', ':\n')
            note = _note(ax, x, y, text, color=S.ONSET_COLORS[case['timing']], ha=ha, va=va)
            if PARTIAL:
                _clear_of_data(ax, note)
        if PARTIAL:
            for ax, text in zip(row, partial_notes):
                ax.annotate(text, (0.5, 0), xycoords='axes fraction', xytext=(0, -13),
                            textcoords='offset points', ha='center', va='top',
                            fontsize=S.FS_NOTE, color=S.NOTE_COLOR, linespacing=1.1)
        _row_label(fig, row[0], case.get('row_label_b', case['row_label']),
                   fontsize=S.FS_TICK)
    c_bottom = C_TOP + 2 * C_H + C_VGAP
    fig.text(0.5, _y(fig, c_bottom + 0.15 + NOTE_BAND), 'Training epoch', ha='center',
             va='top', fontsize=S.FS_LABEL)
    if PENDING:
        n_frames = 2 * len(FRAME_ROLES)
        missing = missing_t + missing_u
        fig.text(0.5, _y(fig, b_bottom + 0.15),
                 f'{PARTIAL_TAG}: '
                 + (f'{missing} of {n_frames} TriMaps not yet computed; ' if missing else '')
                 + 'values not yet computed read "pending" (A) or are counted under '
                   'each panel (B)',
                 ha='center', va='top', fontsize=S.FS_NOTE, color=S.NOTE_COLOR)

    S.case_tag(grid[0][0], -0.62, 1.12, 'T', coords='axes')
    _header(fig, A_HEAD, 'A',
            f'TriMaps of penultimate features, {n_img:,} ImageNet validation images')
    _header(fig, C_TOP - 0.20, 'B', 'Per-epoch readouts')

    S.save(fig, FIGURES, STEM)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    make_figure()


if __name__ == '__main__':
    main()
