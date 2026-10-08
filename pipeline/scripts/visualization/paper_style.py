"""Shared print-size style for the paper figure set.

Every figure is drawn at its final printed width -- the paper's text width,
5.5 in -- so the point sizes below are the sizes that land on paper. The
earlier drafts were drawn 11-23 in wide with 18-22 pt fonts and printed at
4-5 pt once LaTeX scaled them to the text width; drawing at print size is
the only way the numbers here mean what they say.

The visual grammar is shared across figures so a reader learns it once:

* colour is onset timing and nothing else (early / mid / late);
* filled marker + solid line = input-side perturbation, open marker +
  dashed line = target-side perturbation; marker shape = perturbation type;
* black = the unperturbed baseline;
* circled letters tag the three runs every figure follows (T, K, L).
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from arch_runs import SEED_SUFFIX

TEXT_WIDTH = 5.5

FS_TICK = 7.0
FS_LABEL = 7.5
FS_TITLE = 8.0
FS_LETTER = 10.0
FS_LEGEND = 7.0
FS_NOTE = 6.5
MIN_FONT_PT = 6.5

LW_DATA = 1.3
LW_BASELINE = 1.6
LW_REF = 0.7
MS = 4.0
MS_STAR = 7.5

BASELINE_COLOR = '#111111'
REF_COLOR = '#8a8a8a'
NOTE_COLOR = '#555555'
BAND_COLOR = '#e6e4df'
FIRST_EPOCH_SHADE = '#f1f0ec'

TIMINGS = ['early', 'mid', 'late']
ONSET_COLORS = {'early': '#2a78d6', 'mid': '#1baf7a', 'late': '#eb6834'}

# Onset epochs of the one-epoch perturbation, per architecture, shared by the
# figures.
ONSET_EPOCH = {
    'resnet50_imagenet':   {'early': 5, 'mid': 30, 'late': 70},
    'vitb16_imagenet':     {'early': 5, 'mid': 30, 'late': 70},
    'convnext_b_imagenet': {'early': 5, 'mid': 30, 'late': 70},
    'clip_vitb32': {'early': 5, 'mid': 13, 'late': 28},
}
TOTAL_EPOCHS = {
    'resnet50_imagenet': 90, 'vitb16_imagenet': 90,
    'convnext_b_imagenet': 90, 'clip_vitb32': 32,
}
ARCH_LABEL = {
    'resnet50_imagenet':   'RN50',
    'vitb16_imagenet':     'ViT-B/16',
    'convnext_b_imagenet': 'ConvNeXt-B',
    'clip_vitb32': 'CLIP ViT-B/32',
}

INPUT_SIDE = ('image_noise', 'uniform_images')
TARGET_SIDE = ('label_shuffle', 'random_target')
PERT_LABEL = {
    'image_noise':    'image noise',
    'uniform_images': 'uniform images',
    'label_shuffle':  'label shuffle',
    'random_target':  'random target',
}
# (marker, filled). Filled = input-side, open = target-side.
PERT_MARKER = {
    'image_noise':    ('o', True),
    'uniform_images': ('D', True),
    'label_shuffle':  ('o', False),
    'random_target':  ('s', False),
}


# TriMap category colouring, shared by Figs 3 and 4. The five largest
# WordNet categories of the ImageNet classes keep their own colour and the
# other ten share one; the hues avoid the three onset colours.
TRIMAP_CATEGORY_COLORS = {
    'Animal':    '#984ea3',
    'Vehicle':   '#e7298a',
    'Structure': '#e6ab02',
    'Clothing':  '#111111',
    'Food':      '#a65628',
}
TRIMAP_OTHER_COLOR = '#8c8c8c'
TRIMAP_UNCATEGORIZED_COLOR = '#d6d6d6'


def trimap_scatter(ax, coords, categories, s=0.1, alpha=0.65, seed=0):
    """One TriMap frame: uncategorized points underneath, then every
    categorized point in a single pass in a fixed random order. Drawing one
    category at a time would let whichever is drawn last cover a collapsed
    map entirely, misrepresenting the category mix where points overlap."""
    import numpy as np
    from matplotlib.colors import to_rgba_array
    categories = np.asarray(categories)
    under = categories == 'Uncategorized'
    ax.scatter(coords[under, 0], coords[under, 1], s=s,
               c=TRIMAP_UNCATEGORIZED_COLOR, alpha=alpha, linewidths=0,
               rasterized=True)
    idx = np.flatnonzero(~under)
    idx = idx[np.random.default_rng(seed).permutation(len(idx))]
    colors = to_rgba_array([TRIMAP_CATEGORY_COLORS.get(c, TRIMAP_OTHER_COLOR)
                            for c in categories[idx]])
    ax.scatter(coords[idx, 0], coords[idx, 1], s=s, c=colors, alpha=alpha,
               linewidths=0, rasterized=True)


def procrustes_onto(ref, moving):
    """Rotate/reflect `moving` onto `ref` (matched rows) and match its RMS
    spread. The least-squares Procrustes scale would shrink a map that
    corresponds weakly to its reference toward a point."""
    from scipy.linalg import orthogonal_procrustes
    import numpy as np
    mr, mm = ref.mean(0), moving.mean(0)
    a, b = ref - mr, moving - mm
    R, _ = orthogonal_procrustes(b, a)
    b_rot = b @ R
    return np.sqrt(np.sum(a * a) / np.sum(b_rot * b_rot)) * b_rot + mr


def pert_linestyle(pert):
    return '-' if pert in INPUT_SIDE else (0, (3.5, 1.8))


def marker_kw(pert, color, ms=MS, mew=0.9):
    """Line2D/plot kwargs for a perturbation's marker in a given colour."""
    marker, filled = PERT_MARKER[pert]
    size = ms * 0.85 if marker == 'D' else ms
    return dict(marker=marker, markersize=size, markeredgewidth=mew,
                markeredgecolor=color,
                markerfacecolor=color if filled else 'white')


def scatter_kw(pert, color, s=None, lw=0.9):
    """ax.scatter kwargs matching marker_kw."""
    marker, filled = PERT_MARKER[pert]
    size = (MS * (0.85 if marker == 'D' else 1.0)) ** 2 if s is None else s
    return dict(marker=marker, s=size, linewidths=lw, edgecolors=color,
                facecolors=color if filled else 'white')


def apply():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'font.size': FS_LABEL,
        'axes.titlesize': FS_TITLE,
        'axes.titleweight': 'normal',
        'axes.labelsize': FS_LABEL,
        'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_LEGEND,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'axes.labelpad': 2.5,
        'axes.titlepad': 3.5,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.pad': 1.8,
        'ytick.major.pad': 1.8,
        'lines.linewidth': LW_DATA,
        'lines.markersize': MS,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'savefig.bbox': 'standard',
        'figure.dpi': 150,
    })


def onset_handles(with_baseline=True, star=True):
    """Legend handles: black baseline + one colour per onset timing."""
    hs = []
    if with_baseline:
        hs.append(Line2D([], [], color=BASELINE_COLOR, lw=LW_BASELINE,
                         label='unperturbed baseline'))
    for t in TIMINGS:
        kw = dict(marker='*', markersize=MS_STAR, markeredgewidth=0) if star else {}
        hs.append(Line2D([], [], color=ONSET_COLORS[t], lw=LW_DATA,
                         label=f'{t} onset', **kw))
    return hs


def pert_handles(perts=('image_noise', 'uniform_images', 'label_shuffle'),
                 color='#444444', lines=True):
    """Legend handles: marker shape/fill (+ line style) per perturbation."""
    hs = []
    for p in perts:
        side = 'input' if p in INPUT_SIDE else 'target'
        hs.append(Line2D([], [], color=color,
                         lw=LW_DATA if lines else 0,
                         ls=pert_linestyle(p) if lines else 'none',
                         label=f'{PERT_LABEL[p]} ({side}-side)',
                         **marker_kw(p, color)))
    return hs


def case_tag(ax, x, y, letter, coords='data', fontsize=FS_NOTE + 0.5,
             zorder=12, **kw):
    """Circled case letter (T, K, L) at (x, y)."""
    transform = ax.transAxes if coords == 'axes' else ax.transData
    return ax.text(x, y, letter, transform=transform, ha='center',
                   va='center', fontsize=fontsize, fontweight='bold',
                   zorder=zorder,
                   bbox=dict(boxstyle='circle,pad=0.18', facecolor='white',
                             edgecolor='#111111', linewidth=0.7), **kw)


def panel_header(fig, axes, letter, title, pad_pt=3.0, x_axes=None,
                 y_axes=None):
    """Bold panel letter + plain title on one line above a group of axes.

    Placed at the left edge of the group's rendered extent (tick labels
    included) and just above its top, so it cannot collide with them.
    `x_axes` overrides the axes whose left edge anchors the letter;
    `y_axes` (a list) sets the height, so side-by-side panels can share a
    header line.
    """
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [a.get_tightbbox(r).transformed(inv) for a in axes]
    x0 = min(b.x0 for b in boxes)
    if x_axes is not None:
        x0 = x_axes.get_tightbbox(r).transformed(inv).x0
    y_boxes = boxes if y_axes is None else [
        a.get_tightbbox(r).transformed(inv) for a in y_axes]
    y1 = max(b.y1 for b in y_boxes)
    dy = pad_pt / 72.0 / fig.get_figheight()
    t_letter = fig.text(x0, y1 + dy, letter, fontsize=FS_LETTER,
                        fontweight='bold', ha='left', va='baseline')
    fig.canvas.draw()
    bb = t_letter.get_window_extent(r).transformed(inv)
    gap = 3.5 / 72.0 / fig.get_figwidth()
    fig.text(bb.x1 + gap, y1 + dy, title, fontsize=FS_TITLE,
             ha='left', va='baseline')


def check_print_size(fig, name=''):
    """Fail loudly if the canvas is not print width or any text is too small."""
    w = fig.get_figwidth()
    if abs(w - TEXT_WIDTH) > 1e-6:
        raise ValueError(f'{name}: figure is {w:.2f} in wide, expected {TEXT_WIDTH}')
    small = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < MIN_FONT_PT - 1e-6:
            small.append((round(t.get_fontsize(), 2), t.get_text()[:40]))
    if small:
        raise ValueError(f'{name}: {len(small)} text items below '
                         f'{MIN_FONT_PT} pt, e.g. {small[:5]}')
    # Saved without bbox cropping, so anything drawn past the canvas edge
    # would be silently cut off in the PDF.
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    tol = 0.02
    if (bb.x0 < -tol or bb.y0 < -tol or bb.x1 > w + tol
            or bb.y1 > fig.get_figheight() + tol):
        raise ValueError(f'{name}: content extends past the canvas '
                         f'({bb.x0:.2f}, {bb.y0:.2f}, {bb.x1:.2f}, {bb.y1:.2f}) '
                         f'vs {w:.2f} x {fig.get_figheight():.2f} in')


def save(fig, outdir, stem):
    """Save PDF + 300-dpi PNG at the drawn size (no bbox cropping of the
    width, so the PDF drops into LaTeX at exactly 5.5 in). A replicate seed's
    files carry its suffix, so they can sit beside the seed-42 figures."""
    check_print_size(fig, stem)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # A module that sets savefig.bbox='tight' at import time would crop/extend
    # the canvas and break the 5.5 in guarantee.
    with plt.rc_context({'savefig.bbox': 'standard'}):
        for ext in ('pdf', 'png'):
            out = outdir / f'{stem}{SEED_SUFFIX}.{ext}'
            fig.savefig(out, dpi=300)
            print(f'Saved {out}')
