"""Fig 3 -- the representation at the perturbed epoch, drawn at print size.

Definitions are in the paper's figure caption.

A  ImageNet top-1 vs. behavioral alignment at the perturbed epoch, one panel
   per architecture. Each perturbed run is joined to the unperturbed baseline
   at the same epoch (star); the black path joins the baseline across onsets.
B  TriMaps of the 50,000 ImageNet validation images at epoch 70 for the two
   case architectures (baseline + two late perturbations each). Every
   perturbed map is Procrustes-aligned onto its baseline -- TriMap fits each
   model separately, so its axes are arbitrary up to rotation/reflection/scale.
C  Class-gap ratio vs. top-1 retained, and category-gap ratio vs. alignment
   retained, across the 36 perturbed runs; the six runs whose category gap
   falls below 22% of their baseline's are ringed.
D  Every structure-readout correlation with Fisher-z 95% CIs, one row per
   structure metric (F, S, class gap, category gap) and one column per readout:
   zero-order, and partialling out the other metric of its pair (F and S; the
   two gaps) or retained 10-NN class purity.
A second figure (STEM_APPENDIX, appendix) holds the F and S scatters that
were panel C before the gaps replaced them.

Structure metrics come from <results_root>/global_local_structure_analysis.json,
computed on the same 50,000 validation images (penultimate features; CLIP:
768-d CLS): F = Spearman rho between the baseline's and the perturbed model's
16x16 cosine RDMs of WordNet-category centroids; S = cosine silhouette of the
16 categories (sampled); 10-NN purity = fraction of an image's 10 cosine
nearest neighbours from its own class; retained = perturbed / baseline.
The gaps come from pipeline/results/distance_levels_analysis.json (mean-centred
cosine distances within a class d1, between classes of one WordNet category d2,
and between categories d3): class gap (d2 - d1) / d3, category gap
(d3 - d2) / d3, each as a ratio to the baseline's at the same epoch.

Usage:
    python pipeline/scripts/visualization/fig3_storyboard.py
    TP_SEED_SUFFIX=_seed43 python pipeline/scripts/visualization/fig3_storyboard.py  # seed-43 replicate
    TP_SEED_SUFFIX=_seed43 TP_PARTIAL=1 python pipeline/scripts/visualization/fig3_storyboard.py
        # from the seed-43 inputs computed so far; grey notes say what is missing
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_runs import (CLIP_PREFIX, FIGURES, PARTIAL, RESULTS, SEED_SUFFIX, baseline_run,
                       clip_imagenet_probe, imagenet_val_curves_path,
                       pending, rdm1854_corrected_suffix, seeded, structure_analysis_path,
                       widen_for_seed)
import make_fig_cross_architecture as CA
import paper_style as S

S.apply()

_REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGENET_CURVES = imagenet_val_curves_path()
BN_RECALIBRATED_DIR = RESULTS / 'bn_recalibrated_val'
BN_RECALIBRATED_ARCHES = {'resnet50_imagenet'}
CHANCE_FLOOR_JSON = RESULTS / 'spose_chance_floor.json'
STRUCTURE_JSON = structure_analysis_path()
DISTANCE_JSON = RESULTS / f'distance_levels_analysis{SEED_SUFFIX}.json'
STEM = 'Fig3_storyboard_defs_in_caption'
STEM_APPENDIX = 'FigS_fidelity_separability'
SUPERCLASS_JSON = (_REPO_ROOT / 'temporal_perturbations' / 'data'
                   / '_imagenet_superclass_cache.json')
TRIMAP_COORDS_DIR = RESULTS / 'trimap'
TRIMAP_SYNSETS_DIR = RESULTS / 'imagenet_embeddings'

ONSET_EPOCH = {k: S.ONSET_EPOCH[k] for k in
               ('resnet50_imagenet', 'vitb16_imagenet', 'convnext_b_imagenet', CLIP_PREFIX)}
ARCHES = [(k, S.ARCH_LABEL[k]) for k in ONSET_EPOCH]
TIMINGS = S.TIMINGS
PERTS = ['image_noise', 'uniform_images', 'label_shuffle']
ARCH_KEY_TO_CHANCE = {
    'resnet50_imagenet':   'RN50',
    'vitb16_imagenet':     'ViT-B/16',
    'convnext_b_imagenet': 'ConvNeXt-B',
    CLIP_PREFIX:           'CLIP ViT-B/32',
}
# global_local_structure_analysis.json names architectures by these labels.
STRUCTURE_ARCH = {'vitb16_imagenet': 'ViT-B/16', 'convnext_b_imagenet': 'ConvNeXt-B'}

# (architecture, perturbation) of the tagged runs; both are late onsets.
CASES = {('vitb16_imagenet', 'image_noise'): 'T',
         ('convnext_b_imagenet', 'label_shuffle'): 'K'}

TRIMAP_ROWS = [
    ('vitb16_imagenet', f'{baseline_run("vitb16_imagenet")}_epoch70_synsets.json', [
        (None, f'trimap_vitb16_baseline_vs_image_noise_late{SEED_SUFFIX}_epoch70_coords_baseline.npy'),
        ('image_noise', f'trimap_vitb16_baseline_vs_image_noise_late{SEED_SUFFIX}_epoch70_coords_perturbed.npy'),
        ('uniform_images', f'trimap_vitb16_baseline_vs_uniform_images_late{SEED_SUFFIX}_epoch70_coords_perturbed.npy'),
    ]),
    ('convnext_b_imagenet', f'{baseline_run("convnext_b_imagenet")}_epoch70_synsets.json', [
        (None, f'trimap_convnextb_baseline_vs_label_shuffle_late{SEED_SUFFIX}_epoch70_coords_baseline.npy'),
        ('label_shuffle', f'trimap_convnextb_baseline_vs_label_shuffle_late{SEED_SUFFIX}_epoch70_coords_perturbed.npy'),
        ('uniform_images', f'trimap_convnextb_baseline_vs_uniform_images_late{SEED_SUFFIX}_epoch70_coords_perturbed.npy'),
    ]),
]

CATEGORY_COLORS = S.TRIMAP_CATEGORY_COLORS
OTHER_NAMED = 'other named category'
OTHER_COLOR = S.TRIMAP_OTHER_COLOR
UNCATEGORIZED_COLOR = S.TRIMAP_UNCATEGORIZED_COLOR

W = S.TEXT_WIDTH
# Partial mode adds a strip under A and one under C/D for the notes on what is missing.
PARTIAL_STRIP = 0.14 if PARTIAL else 0.0
H = 4.56 + 2 * PARTIAL_STRIP
A_TOP, A_HEIGHT = 0.66, 0.62
MAP_HEIGHT = 0.40
# C and D are set larger than the rest of the figure, which holds them to the
# print minimum. C's axes run from the band top to C_BOTTOM above the canvas
# edge; D's start lower, under its legend and column titles.
FS_CD_TICK, FS_CD_LABEL, FS_CD_NOTE = 8.0, 8.5, 7.5
C_X, C_W = [0.50, 1.84], 0.84
C_BOTTOM = 0.36
D_HEAD_X, D_X, D_W, D_GAP = 2.80, 3.62, 0.80, 0.08
# D's axes start under its column titles; its key takes the line where an x-axis
# label would go (the header carries n).
D_TOP_OFFSET = 0.13
# The six runs keeping the least alignment (0.01-0.13 of baseline) are exactly
# those whose category gap falls below this fraction of their baseline's; the
# next lowest is 0.37 (Section 4.2).
GAP_THRESHOLD = 0.22
GAP_BELOW_THRESHOLD = 6


def _run_name(key, pert, timing):
    return seeded(f'{key}_{pert}_{timing}')


def _clip_ood_probe_acc(run_name, epoch):
    """CLIP's ImageNet probe top-1 at `epoch` (current probe recipe only), or None if absent."""
    try:
        return clip_imagenet_probe(run_name, epoch)['imagenet_val']['top1_accuracy']
    except FileNotFoundError:
        return None


def _bn_recalibrated_acc(run_name, epoch):
    p = BN_RECALIBRATED_DIR / f'{run_name}_epoch{epoch}.json'
    if not p.is_file():
        return None
    with p.open() as f:
        return json.load(f)['bn_recalibrated']['val_acc1']


def _skip(warning, what):
    """Drop a missing value: with a warning, as always, or recorded as pending in partial mode."""
    if PARTIAL:
        pending(what)
    else:
        print(f'  WARNING: {warning}')


def _drawn(values, k):
    """values[k]; partial mode draws a missing value (already recorded as pending) as not yet computed."""
    return values.get(k) if PARTIAL else values[k]


def _top1_what(key):
    return 'ImageNet probe top-1' if key == CLIP_PREFIX else 'ImageNet top-1'


def _alignment_suffix(key):
    s = rdm1854_corrected_suffix(key)
    if key in BN_RECALIBRATED_ARCHES:
        return '_bn_recal' + s
    return s


def load_alignment_at_onset():
    """Returns {(arch_key, condition, timing): rsa_score}.

    condition is 'baseline' or a perturbation type name. ResNet-50 reads the
    BatchNorm-recalibrated variant.
    """
    out = {}
    for key, _ in ARCHES:
        suffix = _alignment_suffix(key)
        base_traj = CA.load_traj(baseline_run(key), suffix)
        if base_traj is None:
            _skip(f'no baseline alignment for {key}',
                  f'{baseline_run(key)} behavioral alignment: all epochs')
            continue
        b_ep, b_rsa, _ = base_traj
        b_lookup = dict(zip(b_ep.astype(int).tolist(), b_rsa.tolist()))
        for t in TIMINGS:
            onset = ONSET_EPOCH[key][t]
            if onset in b_lookup:
                out[(key, 'baseline', t)] = b_lookup[onset]
            else:
                _skip(f'no baseline alignment at epoch {onset} for {key}',
                      f'{baseline_run(key)} behavioral alignment epoch {onset}')
            for pert in PERTS:
                run = _run_name(key, pert, t)
                tr = CA.load_traj(run, suffix)
                if tr is None:
                    _skip(f'no alignment trajectory for {run}',
                          f'{run} behavioral alignment epoch {onset}')
                    continue
                p_ep, p_rsa, _ = tr
                p_lookup = dict(zip(p_ep.astype(int).tolist(), p_rsa.tolist()))
                if onset in p_lookup:
                    out[(key, pert, t)] = p_lookup[onset]
                else:
                    _skip(f'no epoch-{onset} alignment for {run}',
                          f'{run} behavioral alignment epoch {onset}')
    return out


def load_accuracy_at_onset():
    """Returns {(arch_key, condition, timing): accuracy_0_to_1}.

    BN-recalibrated for ResNet-50, imagenet_val_curves for ViT/ConvNeXt, and
    the ImageNet linear probe for CLIP ViT-B/32 (it has no classifier head).
    """
    curves = {}
    if IMAGENET_CURVES.is_file():
        with IMAGENET_CURVES.open() as f:
            curves = json.load(f)['runs']

    out = {}
    for key, _ in ARCHES:
        for t in TIMINGS:
            onset = ONSET_EPOCH[key][t]
            for cond in ['baseline'] + PERTS:
                run = baseline_run(key) if cond == 'baseline' else _run_name(key, cond, t)
                if key == CLIP_PREFIX:
                    acc = _clip_ood_probe_acc(run, onset)
                elif key in BN_RECALIBRATED_ARCHES:
                    acc = _bn_recalibrated_acc(run, onset)
                else:
                    acc = curves.get(run, {}).get('val_acc1', {}).get(str(onset))
                if acc is None:
                    _skip(f'no accuracy at epoch {onset} for {run}',
                          f'{run} {_top1_what(key)} epoch {onset}')
                    continue
                out[(key, cond, t)] = acc
    return out


def load_random_init_alignment():
    """Returns {arch_key: mean rho} over the 5 random-init seeds."""
    with CHANCE_FLOOR_JSON.open() as f:
        raw = json.load(f)
    return {key: raw[name]['mean'] for key, name in ARCH_KEY_TO_CHANCE.items()
            if name in raw}


def load_structure_rows():
    try:
        with STRUCTURE_JSON.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        pending(f'{STRUCTURE_JSON.name}: all runs')
        return []
    rows = [r for r in data if r.get('pert') != 'baseline']
    # The analysis skips runs it has no embeddings for, and panel D's n would
    # shrink silently.
    have = {(r['arch_key'], r['pert'], r['timing']) for r in rows}
    for k, p, t in sorted({(k, p, t) for k in ONSET_EPOCH for p in PERTS for t in TIMINGS} - have):
        _skip(f'no structure row for {_run_name(k, p, t)}',
              f'{_run_name(k, p, t)} structure metrics epoch {ONSET_EPOCH[k][t]}')
    kept = []
    for r in rows:
        if r.get('acc_retained') is None or r.get('align_retained') is None:
            run = _run_name(r['arch_key'], r['pert'], r['timing'])
            # The ratios are filled in when the analysis is re-run after the accuracy lands.
            _skip(f"dropping structure row for {run} (acc_retained "
                  f"{r.get('acc_retained')}, align_retained {r.get('align_retained')})",
                  f"{run} structure row "
                  f"{'acc_retained' if r.get('acc_retained') is None else 'align_retained'} "
                  f"epoch {ONSET_EPOCH[r['arch_key']][r['timing']]}")
        else:
            kept.append(r)
    return kept


def _gaps(rec):
    """(class gap, category gap) of one distance-levels record."""
    d3 = rec['centered_between_category']
    return ((rec['centered_between_class'] - rec['centered_within_class']) / d3,
            (d3 - rec['centered_between_class']) / d3)


def load_gap_ratios(rows):
    """(class-gap ratio, category-gap ratio) for each of `rows`, each gap as the
    run's over its baseline's at the same epoch; None for a run the analysis has
    not scored yet (partial mode)."""
    try:
        recs = json.loads(DISTANCE_JSON.read_text())
    except FileNotFoundError:
        pending(f'{DISTANCE_JSON.name}: all runs')
        return [None] * len(rows)
    base = {(r['arch'], r['timing']): _gaps(r) for r in recs if r['pert'] == 'baseline'}
    runs = {(r['arch'], r['timing'], r['pert']): _gaps(r) for r in recs if r['pert'] != 'baseline'}
    out = []
    for row in rows:
        k = (row['arch'], row['timing'], row['pert'])
        if k not in runs or k[:2] not in base:
            pending(f"{_run_name(row['arch_key'], row['pert'], row['timing'])} distance levels")
            out.append(None)
            continue
        (cg, kg), (cg_base, kg_base) = runs[k], base[k[:2]]
        out.append((cg / cg_base, kg / kg_base))
    return out


def _partial_corr(x, y, z):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(rx, ry)


def _fisher_ci(r, n, n_covariates=0):
    se = 1.0 / np.sqrt(n - 3 - n_covariates)
    z = np.arctanh(r)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


# Layout helpers (inches from the top-left corner)

def _axes(fig, left, top, width, height):
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    return fig.add_axes([left / fw, 1 - (top + height) / fh, width / fw, height / fh])


def _text(fig, x, y, s, **kw):
    kw.setdefault('fontsize', S.FS_NOTE)
    kw.setdefault('va', 'top')
    return fig.text(x / fig.get_figwidth(), 1 - y / fig.get_figheight(), s, **kw)


def _header(fig, x, y, letter, title):
    """Panel letter + (possibly two-line) title, top-aligned at (x, y) in."""
    _text(fig, x, y - 0.01, letter, fontsize=S.FS_LETTER, fontweight='bold')
    _text(fig, x + 0.17, y, title, fontsize=S.FS_TITLE, linespacing=1.2)


def _fmt_acc(v):
    return f'{v:.0f}' if v >= 10 else f'{v:.1f}'


N_RUNS = len(ONSET_EPOCH) * len(PERTS) * len(TIMINGS)
PARTIAL_TAG = f'Seed {SEED_SUFFIX.removeprefix("_seed")}, partial'


def _not_yet_computed(ax, text='not yet computed'):
    """Mark an axes whose inputs have not landed; keeps the panel's footprint."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=S.FS_NOTE, color=S.NOTE_COLOR, linespacing=1.1)


def _panel_a_note(key, arch_label, alignment, accuracy):
    """How many of the architecture's perturbed runs panel A leaves out, per missing readout
    (the run's own value or its same-epoch baseline's); None when it draws all of them."""
    runs = [(p, t) for t in TIMINGS for p in PERTS]
    parts = []
    for what, values in ((_top1_what(key).removeprefix('ImageNet '), accuracy),
                         ('alignment', alignment)):
        k = sum((key, 'baseline', t) not in values or (key, p, t) not in values
                for p, t in runs)
        if k:
            parts.append(f'{what} not yet computed for {k} of {len(runs)} runs')
    return f'{arch_label} {", ".join(parts)}' if parts else None


def _num(v, fmt):
    return format(v, fmt).replace('-', '\u2212')


# Key

def draw_key(fig):
    star = [Line2D([], [], ls='none', marker='*', markersize=S.MS_STAR,
                   markerfacecolor='#bbbbbb', markeredgecolor='#111111',
                   markeredgewidth=0.5, label='unperturbed baseline, same epoch')]
    onsets = [Line2D([], [], color=S.ONSET_COLORS[t], lw=S.LW_DATA,
                     label=f'{t} onset') for t in TIMINGS]
    fig.legend(handles=star + onsets, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1 - 0.02 / H), handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0)
    fig.legend(handles=S.pert_handles(PERTS, lines=False), loc='upper center',
               ncol=3, bbox_to_anchor=(0.5, 1 - 0.18 / H), handlelength=1.0,
               columnspacing=1.6, handletextpad=0.4, borderaxespad=0)


# ---------------------------------------------------------------------------
# Panel A

def plot_panel_a(fig, top, alignment, accuracy, random_init):
    left, width, gap, height = 0.60, 0.98, 0.27, A_HEIGHT
    axes = []
    for i, (key, arch_label) in enumerate(ARCHES):
        ax = _axes(fig, left + i * (width + gap), top, width, height)
        axes.append(ax)

        base_pts = []
        for t in TIMINGS:
            b_acc = accuracy.get((key, 'baseline', t))
            b_al = alignment.get((key, 'baseline', t))
            if b_acc is None or b_al is None:
                continue
            color = S.ONSET_COLORS[t]
            bx = b_acc * 100
            base_pts.append((bx, b_al))
            for pert in PERTS:
                p_acc = accuracy.get((key, pert, t))
                p_al = alignment.get((key, pert, t))
                if p_acc is None or p_al is None:
                    continue
                px = p_acc * 100
                case = CASES.get((key, pert)) if t == 'late' else None
                ax.plot([bx, px], [b_al, p_al], color=color,
                        lw=1.3 if case else 0.5, alpha=1.0 if case else 0.55,
                        zorder=2.5 if case else 2, solid_capstyle='round')
                ax.scatter(px, p_al, zorder=4, **S.scatter_kw(pert, color))
                if case:
                    off = (-9, 1) if case == 'T' else (9, 5)
                    ax.annotate(case, (px, p_al), xytext=off,
                                textcoords='offset points', ha='center',
                                va='center', fontsize=S.FS_NOTE + 0.5,
                                fontweight='bold', zorder=12,
                                bbox=dict(boxstyle='circle,pad=0.18',
                                          facecolor='white', edgecolor='#111111',
                                          linewidth=0.7))
            ax.scatter(bx, b_al, marker='*', s=S.MS_STAR ** 2 * 1.1,
                       facecolors=color, edgecolors='#111111', linewidths=0.5,
                       zorder=6)
        if base_pts:
            bxs, bys = zip(*base_pts)
            ax.plot(bxs, bys, color=S.BASELINE_COLOR, lw=2.0,
                    zorder=3, solid_capstyle='round')

        if key in random_init:
            y0 = random_init[key]
            ax.axhline(y0, color=S.REF_COLOR, lw=0.8, ls=(0, (1, 1.6)), zorder=1)
            # One inline label teaches the line; the caption defines it for all.
            if i == 0:
                ax.text(0.98, y0, 'random init', transform=ax.get_yaxis_transform(),
                        ha='right', va='bottom', fontsize=S.FS_NOTE,
                        color=S.NOTE_COLOR)

        ax.set_xlim(-5, None)
        ax.margins(y=0.10)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.set_title(arch_label, fontsize=S.FS_LABEL)
        ax.set_xlabel('ImageNet probe top-1 (%)' if key == CLIP_PREFIX
                      else 'ImageNet top-1 (%)')
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0].set_ylabel('Behavioral\nalignment (ρ)')
    return axes


# Panel B

def _categories(synset_ids):
    with SUPERCLASS_JSON.open() as f:
        cat = json.load(f)
    labels = []
    for s in synset_ids:
        c = cat.get(s, 'Uncategorized')
        if c != 'Uncategorized' and c not in CATEGORY_COLORS:
            c = OTHER_NAMED
        labels.append(c)
    return np.array(labels)


def _other_named_stats(synset_ids):
    """Number of named categories drawn in OTHER_COLOR, and the largest image share of any one."""
    with SUPERCLASS_JSON.open() as f:
        cat = json.load(f)
    raw = np.array([cat.get(s, 'Uncategorized') for s in synset_ids])
    other = {c for c in raw if c != 'Uncategorized' and c not in CATEGORY_COLORS}
    return len(other), max(np.mean(raw == c) for c in other)


def _load_trimap(coords_file, run):
    try:
        return np.load(TRIMAP_COORDS_DIR / coords_file)
    except FileNotFoundError:
        pending(f'{run} TriMap epoch 70')
        return None


def _structure_lookup(rows):
    return {(r['arch'], r['pert'], r['timing']): r for r in rows}


def plot_panel_b(fig, top, alignment, accuracy, structure):
    """All six maps in one row: ViT-B/16's three, then ConvNeXt-B's three. Also returns the
    number of maps not computed yet."""
    width, gap, group_gap, height = 0.83, 0.06, 0.18, MAP_HEIGHT
    left = (W - (6 * width + 4 * gap + group_gap)) / 2
    maps_top = top + 0.31
    first_axes = []
    all_synsets = []
    n_missing = 0
    for gi, (key, syn_file, maps) in enumerate(TRIMAP_ROWS):
        with (TRIMAP_SYNSETS_DIR / syn_file).open() as f:
            synsets = json.load(f)
        cats = _categories(synsets)
        all_synsets += synsets
        base_coords = _load_trimap(maps[0][1], baseline_run(key))
        n_missing += base_coords is None
        if base_coords is not None:
            # Shared limits per architecture, from the baseline's central 99.6% so
            # a few outlying points don't shrink the clouds.
            lo = np.percentile(base_coords, 0.2, axis=0)
            hi = np.percentile(base_coords, 99.8, axis=0)
            span = hi - lo
            lo, hi = lo - 0.04 * span, hi + 0.04 * span
        g_left = left + gi * (3 * width + 2 * gap + group_gap)
        for ci, (pert, coords_file) in enumerate(maps):
            ax = _axes(fig, g_left + ci * (width + gap), maps_top, width, height)
            if ci == 0:
                first_axes.append(ax)
            coords, absent = base_coords, 'not yet computed'
            if pert is not None:
                moving = _load_trimap(coords_file, _run_name(key, pert, 'late'))
                n_missing += moving is None
                if moving is None:
                    coords = None
                elif base_coords is None:
                    # A perturbed map is drawn only aligned onto its baseline.
                    coords, absent = None, 'drawn once the\nbaseline map lands'
                else:
                    coords = S.procrustes_onto(base_coords, moving)
            if coords is None:
                _not_yet_computed(ax, absent)
            else:
                S.trimap_scatter(ax, coords, cats, s=0.05)
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
            ax.set_xticks([])
            ax.set_yticks([])
            frame = S.BASELINE_COLOR if pert is None else S.ONSET_COLORS['late']
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(frame)
                sp.set_linewidth(0.8 if pert is None else 1.1)

            if pert is None:
                ax.set_title('baseline', fontsize=S.FS_LABEL, pad=2)
            else:
                ax.set_title(S.PERT_LABEL[pert], fontsize=S.FS_LABEL, pad=2)
                case = CASES.get((key, pert))
                if case:
                    S.case_tag(ax, 0.88, 0.80, case, coords='axes')
            k = (key, pert or 'baseline', 'late')
            acc, al = _drawn(accuracy, k), _drawn(alignment, k)
            acc_txt = 'pending' if acc is None else f'{_fmt_acc(acc * 100)}%'
            al_txt = 'pending' if al is None else f'{al:.3f}'
            ax.text(0.5, -0.05, f'top-1 {acc_txt}\nalignment ρ {al_txt}',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=S.FS_TICK, linespacing=1.15)
        g_right = g_left + 3 * width + 2 * gap
        _text(fig, (g_left + g_right) / 2, top + 0.02, S.ARCH_LABEL[key],
              ha='center', fontsize=S.FS_LABEL)
        fig.add_artist(Line2D([g_left / W, g_right / W],
                              [1 - (top + 0.15) / H] * 2, color='#111111',
                              lw=0.6, transform=fig.transFigure))

    n_other, max_share = _other_named_stats(all_synsets)
    handles = []
    for cat in list(CATEGORY_COLORS) + [OTHER_NAMED, 'Uncategorized']:
        color = {'Uncategorized': UNCATEGORIZED_COLOR,
                 OTHER_NAMED: OTHER_COLOR}.get(cat, CATEGORY_COLORS.get(cat))
        name = {OTHER_NAMED: f'{n_other} smaller categories '
                             f'(< {int(np.ceil(100 * max_share))}% each)',
                'Uncategorized': 'uncategorized'}.get(cat, cat.lower())
        handles.append(Line2D([], [], ls='none', marker='o', markersize=3.6,
                              markerfacecolor=color, markeredgewidth=0,
                              label=name))
    leg_y = maps_top + height + 0.30
    fig.legend(handles=handles, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1 - leg_y / H), handlelength=0.6,
               columnspacing=0.9, handletextpad=0.25, borderaxespad=0,
               fontsize=S.FS_LEGEND)
    return first_axes, leg_y, n_missing


# ---------------------------------------------------------------------------
# Panels C and D, and the appendix figure
# ---------------------------------------------------------------------------

def _case_of(row):
    """'T' / 'K' for the two case runs (both late onsets), else None."""
    return CASES.get((row['arch_key'], row['pert'])) if row['timing'] == 'late' else None


def _cd_type(ax):
    """C, D and the appendix figure are set larger than the rest of the figure."""
    ax.tick_params(labelsize=FS_CD_TICK)
    ax.xaxis.label.set_size(FS_CD_LABEL)
    ax.yaxis.label.set_size(FS_CD_LABEL)


def _scatter_runs(ax, rows, x, y, tag_offsets):
    for r, xi, yi in zip(rows, x, y):
        ax.scatter(xi, yi, zorder=3, **S.scatter_kw(r['pert'], S.ONSET_COLORS[r['timing']]))
        case = _case_of(r)
        if case:
            ax.annotate(case, (xi, yi), xytext=tag_offsets[case], textcoords='offset points',
                        ha='center', va='center', fontsize=FS_CD_NOTE, fontweight='bold',
                        zorder=12, bbox=dict(boxstyle='circle,pad=0.18', facecolor='white',
                                             edgecolor='#111111', linewidth=0.7))


def _log_axis(ax, lim, ticks):
    ax.set_xscale('log')
    ax.set_xlim(*lim)
    ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plt.FixedFormatter([f'{t:g}' for t in ticks]))
    ax.xaxis.set_minor_locator(plt.NullLocator())


def plot_panel_c(fig, top, rows, gaps):
    """Class-gap ratio vs. top-1 ratio, and category-gap ratio vs. alignment ratio,
    both gaps on a log axis (the category-gap ratio spans 0.01 to 4.4). The region
    below GAP_THRESHOLD of the baseline's category gap, which holds exactly the six
    runs keeping the least alignment, is shaded. Axes start where D's do, under a
    title with the two correlations."""
    have = [(r, g) for r, g in zip(rows, gaps) if g is not None]
    top += D_TOP_OFFSET
    height = H - PARTIAL_STRIP - C_BOTTOM - top
    specs = [
        (0, 'acc_retained', 'Class-gap ratio', 'Top-1 ratio', (0.09, 1.5), [0.1, 0.3, 1],
         {'T': (-10, 4), 'K': (3, 11)}),
        (1, 'align_retained', 'Category-gap ratio', 'Alignment ratio', (0.004, 8.0),
         [0.01, 0.1, 1], {'T': (-13, 0), 'K': (3, 10)}),
    ]
    for i, (gi, yk, xl, yl, xlim, xticks, offsets) in enumerate(specs):
        ax = _axes(fig, C_X[i], top, C_W, height)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl, labelpad=2)
        _cd_type(ax)
        if not have:
            ax.set_xticks([])
            ax.set_yticks([])
            _not_yet_computed(ax)
            continue
        runs = [r for r, _ in have]
        x = np.array([g[gi] for _, g in have])
        y = np.array([r[yk] for r in runs])
        ax.axhline(1, color=S.REF_COLOR, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
        ax.axvline(1, color=S.REF_COLOR, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
        if gi == 1:
            below = x < GAP_THRESHOLD
            if not PARTIAL and below.sum() != GAP_BELOW_THRESHOLD:
                raise ValueError(f'{below.sum()} runs below {GAP_THRESHOLD:.0%} of the '
                                 f'baseline category gap; the text names {GAP_BELOW_THRESHOLD}')
            ax.axvspan(1e-9, GAP_THRESHOLD, color='#e6e6e6', lw=0, zorder=0)
            ax.axvline(GAP_THRESHOLD, color='#111111', lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.text(0.04, 0.97, f'{below.sum()} runs\n< {GAP_THRESHOLD:.0%}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=FS_CD_NOTE,
                    linespacing=1.15)
            print(f'C: {below.sum()} runs below {GAP_THRESHOLD:.0%}: '
                  + ', '.join(f"{r['arch']} {r['timing']} {r['pert']} "
                              f"({xi:.2f}, alignment {yi:.2f})"
                              for r, xi, yi, b in zip(runs, x, y, below) if b)
                  + f'; next lowest {x[~below].min():.2f}')
        _scatter_runs(ax, runs, x, y, offsets)
        r_val, p_val = stats.pearsonr(x, y)
        rs_val, ps_val = stats.spearmanr(x, y)
        ax.set_title(f'r = {r_val:.2f}, $r_\\mathrm{{s}}$ = {rs_val:.2f}',
                     fontsize=FS_CD_NOTE, pad=3)
        print(f'C: {yl} vs {xl}: Pearson r = {r_val:.3f} (p = {p_val:.2g}), '
              f'Spearman r_s = {rs_val:.3f} (p = {ps_val:.2g}), n = {len(x)} runs')
        _log_axis(ax, widen_for_seed(xlim, x.min(), x.max(), f'C {xl}'), xticks)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.margins(y=0.08)


D_ANALYSES = [
    ('r', '#111111', '#111111', 0),
    ('partial r | pair', '#8a8a8a', '#8a8a8a', 1),
    ('partial r | retained purity', 'white', '#111111', 1),
]


def plot_panel_d(fig, top, rows, gaps):
    """One row per structure metric (F, S, class gap, category gap), one column per
    readout (top-1 ratio, alignment ratio): zero-order Pearson r, and partial r
    controlling for the other metric of the pair or for retained 10-NN class purity
    (perturbed / baseline, the scale of the readouts), each with
    its Fisher-z 95% CI. Returns the axes pair and the run count(s) for the header."""
    have = [(r, g) for r, g in zip(rows, gaps) if g is not None]
    gapped = [r for r, _ in have]
    col = lambda rs, k: np.array([r[k] for r in rs])
    cls = np.array([g[0] for _, g in have])
    cat = np.array([g[1] for _, g in have])
    metrics = [
        ('F', rows, col(rows, 'category_rdm_rho'), col(rows, 'category_silhouette_pert')),
        ('S', rows, col(rows, 'category_silhouette_pert'), col(rows, 'category_rdm_rho')),
        ('class gap', gapped, cls, cat),
        ('category gap', gapped, cat, cls),
    ]
    readouts = [('Top-1 ratio', 'acc_retained'), ('Alignment ratio', 'align_retained')]
    offsets = [-0.24, 0.0, 0.24]
    d_top = top + D_TOP_OFFSET
    height = H - PARTIAL_STRIP - C_BOTTOM - d_top
    axes, ci_ends, n_runs = [], [], set()
    for ci, (title, yk) in enumerate(readouts):
        ax = _axes(fig, D_X + ci * (D_W + D_GAP), d_top, D_W, height)
        axes.append(ax)
        for ri, (name, rs, x, other) in enumerate(metrics):
            if len(rs) < 5:
                continue
            y = col(rs, yk)
            knn = col(rs, 'knn_purity_pert') / col(rs, 'knn_purity_base')
            n_runs.add(len(rs))
            results = [stats.pearsonr(x, y)[0], _partial_corr(x, y, other)[0],
                       _partial_corr(x, y, knn)[0]]
            for r_val, (label, fc, ec, ncov), dy in zip(results, D_ANALYSES, offsets):
                lo, hi = _fisher_ci(r_val, len(rs), ncov)
                ci_ends += [lo, hi]
                ax.plot([lo, hi], [ri + dy] * 2, color='#111111', lw=0.8, zorder=2)
                ax.plot(r_val, ri + dy, 'o', markersize=S.MS, markerfacecolor=fc,
                        markeredgecolor=ec, markeredgewidth=0.8, zorder=3,
                        label=label if (ri, ci) == (0, 0) else None)
            print(f'D: {name:12s} vs {title:15s} r = {results[0]:+.2f}, '
                  f'partial | other of pair = {results[1]:+.2f}, '
                  f'partial | retained 10-NN purity = {results[2]:+.2f} (n = {len(rs)})')
        ax.axvline(0, color=S.REF_COLOR, lw=0.7, zorder=1)
        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([m[0] for m in metrics] if ci == 0 else [])
        ax.tick_params(axis='y', length=0)
        ax.set_title(title, fontsize=FS_CD_LABEL, pad=3)
        _cd_type(ax)
        ax.tick_params(axis='y', labelsize=FS_CD_LABEL)
    lim = widen_for_seed((-0.75, 1.0), min(ci_ends, default=0), max(ci_ends, default=0),
                         'D x-axis')
    for ax in axes:
        ax.set_xlim(*lim)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xticklabels(['−0.5', '0', '0.5'])
    n_txt = (f'n = {n_runs.pop()} runs' if len(n_runs) == 1
             else f'n = {min(n_runs, default=0)}-{max(n_runs, default=0)} runs')
    return axes, n_txt


def plot_d_legend(fig, x, axes):
    """D's key in one row along the bottom edge, from x inches."""
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return
    fig.legend(handles, labels, loc='lower left', ncol=3,
               bbox_to_anchor=(x / W, (PARTIAL_STRIP + 0.02) / H), handlelength=0.8,
               handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0, fontsize=FS_CD_NOTE)


APP_H = 2.25


def make_appendix_figure(rows):
    """The F and S scatters that were Figure 3C, with their least-squares fits."""
    fig = plt.figure(figsize=(W, APP_H))
    fig.legend(handles=S.onset_handles(with_baseline=False, star=False),
               loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1 - 0.02 / APP_H),
               handlelength=1.8, columnspacing=1.4, borderaxespad=0, fontsize=FS_CD_NOTE)
    fig.legend(handles=S.pert_handles(PERTS, lines=False), loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1 - 0.20 / APP_H), handlelength=1.0, columnspacing=1.6,
               handletextpad=0.4, borderaxespad=0, fontsize=FS_CD_NOTE)
    col = lambda k: np.array([r[k] for r in rows])
    specs = [
        ('A', 'Top-1 ratio vs. category fidelity F', col('category_rdm_rho'),
         col('acc_retained'), 'Category fidelity F', 'Top-1 ratio',
         {'T': (-9, 6), 'K': (9, 4)}, 0.80),
        ('B', 'Alignment ratio vs. category separability S', col('category_silhouette_pert'),
         col('align_retained'), 'Category separability S', 'Alignment ratio',
         {'T': (-10, -2), 'K': (9, -3)}, 0.97),
    ]
    top, bottom = 0.70, 0.40
    for i, (letter, title, x, y, xl, yl, offsets, r_y) in enumerate(specs):
        left = 0.62 + i * 2.75
        ax = _axes(fig, left, top, 1.95, APP_H - top - bottom)
        _header(fig, left - 0.57, top - 0.22, letter, title)
        _scatter_runs(ax, rows, x, y, offsets)
        r_val, p_val = stats.pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, slope * xs + intercept, color='#333333', lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.text(0.04, r_y, f'r = {r_val:.2f}', transform=ax.transAxes, ha='left',
                va='top', fontsize=FS_CD_NOTE)
        print(f'appendix {letter}: {yl} vs {xl}: Pearson r = {r_val:.3f}, p = {p_val:.4f}')
        ax.axhline(1, color=S.REF_COLOR, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        _cd_type(ax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.margins(0.08)
    S.save(fig, FIGURES, STEM_APPENDIX)
    plt.close(fig)


# Compose

def make_figure():
    alignment = load_alignment_at_onset()
    accuracy = load_accuracy_at_onset()
    random_init = load_random_init_alignment()
    rows = load_structure_rows()
    structure = _structure_lookup(rows)
    print(f'{len(alignment)} alignment values, {len(accuracy)} accuracy values, '
          f'{len(rows)} structure rows')

    fig = plt.figure(figsize=(W, H))
    draw_key(fig)

    axes_a = plot_panel_a(fig, A_TOP, alignment, accuracy, random_init)
    S.panel_header(fig, axes_a, 'A',
                   'Top-1 vs. alignment at the perturbed epoch')

    a_notes = [n for n in (_panel_a_note(k, lbl, alignment, accuracy) for k, lbl in ARCHES)
               if n] if PARTIAL else []
    if a_notes:
        _text(fig, W - 0.05, A_TOP + A_HEIGHT + 0.33,
              f'{PARTIAL_TAG}: {"; ".join(a_notes)} (not drawn)',
              ha='right', color=S.NOTE_COLOR)

    b_top = A_TOP + A_HEIGHT + 0.57 + PARTIAL_STRIP
    _, leg_y, b_missing = plot_panel_b(fig, b_top, alignment, accuracy, structure)
    _header(fig, 0.05, b_top - 0.15, 'B',
            'TriMaps of the 50,000 ImageNet validation images at epoch 70 '
            '(late onset)')

    cd_top = leg_y + 0.38
    gaps = load_gap_ratios(rows)
    plot_panel_c(fig, cd_top, rows, gaps)
    d_axes, n_txt = plot_panel_d(fig, cd_top, rows, gaps)
    head_y = cd_top - 0.20
    _header(fig, 0.05, head_y, 'C', 'Class gap vs. top-1; category gap vs. alignment')
    _header(fig, D_HEAD_X, head_y, 'D', f'Pearson and partial r, 95% CIs ({n_txt})')
    plot_d_legend(fig, D_HEAD_X + 0.25, d_axes)
    notes = ([f'B: {b_missing} of {len(TRIMAP_ROWS) * 3} TriMaps not yet computed']
             if b_missing else [])
    n_gaps = sum(g is not None for g in gaps)
    if PARTIAL and min(len(rows), n_gaps) < N_RUNS:
        notes.append(f'C and D use n = {min(len(rows), n_gaps)}-{len(rows)} of {N_RUNS} runs; '
                     f'the rest are not yet computed')
    if notes:
        _text(fig, 0.05, H - PARTIAL_STRIP + 0.04, f'{PARTIAL_TAG}: ' + '; '.join(notes),
              color=S.NOTE_COLOR)

    S.save(fig, FIGURES, STEM)
    plt.close(fig)
    if rows:
        make_appendix_figure(rows)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    make_figure()


if __name__ == '__main__':
    main()
