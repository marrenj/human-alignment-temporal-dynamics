"""Shared data loader for the neural FR-RSA figures (Fig_neural_*).

Reads the per-run FR-RSA result JSONs written by the layer-reweighted RSA suites
(``inference_results_{nod,things}_frrsa_epoch{E}.json``, one block per ROI) and
exposes tidy per-architecture trajectories and perturbation deltas. Those untagged
files use the legacy THINGS preprocessing; ``corrected=True`` on the baseline loaders
reads each family's corrected files instead (``arch_runs.frrsa_corrected_tag``).

Files live in ``<results_root>/frrsa/<run>/`` (``arch_runs.frrsa_dir``).

Score convention: we plot the ``insample_score`` (the headline; ``scoring="insample"``
was used for the sweep). Absolute levels are positively biased and the bias grows
with layer count, so RAW cross-architecture magnitude is confounded — but
(a) the temporal SHAPE within an architecture and (b) the perturbation Δ
(perturbed − baseline at matched epoch, same layer count) are both depth-fair.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np

from arch_runs import CLIP_PREFIX, baseline_run, frrsa_corrected_tag, frrsa_dir, seeded

# Canonical ordering / labels; must match make_fig_cross_architecture.py.
# (dir_prefix, display label, colorblind-safe line color)
ARCHES = [
    ("resnet50_imagenet",   "RN50",           "#5B7FA5"),  # house blue
    ("vitb16_imagenet",     "ViT-B/16",        "#E8873C"),  # house orange
    ("convnext_b_imagenet", "ConvNeXt-B",      "#4C9F70"),  # green
    (CLIP_PREFIX,           "CLIP-ViT-B/32",   "#8C6BB1"),  # purple
]
ARCH_LABEL = {p: l for p, l, _ in ARCHES}
ARCH_COLOR = {p: c for p, _, c in ARCHES}

# Retained for callers that need the three ImageNet-supervised architectures
# alone. Every neural figure now includes CLIP -- its FR-RSA sweep is complete.
ARCHES_NO_CLIP = [a for a in ARCHES if a[0] != CLIP_PREFIX]

PERTS = [
    ("label_shuffle",  "Label shuffle"),
    ("image_noise",    "Image noise"),
    ("uniform_images", "Blank grayscale"),
]
TIMINGS = ["early", "mid", "late"]

DATASETS = [("nod", "NOD (14 ROIs)"), ("things", "THINGS-fMRI (27 ROIs)")]

# NOD ROIs, ordered early -> late visual hierarchy (HCP-MMP names).
NOD_ROI_ORDER = ["V1", "V2", "V3", "V4", "V8", "LO1", "LO2", "LO3",
                 "PIT", "FFC", "VMV1", "VMV2", "VMV3", "VVC"]

# Nominal full-schedule length per architecture, keyed by run prefix. Same
# numbers and same `100 * epoch / (total - 1)` convention as
# figS_baseline_behavioral_alignment.TOTAL_EPOCHS (which keys by display label), so a
# "% of training" axis means the same thing in every figure in the paper.
TOTAL_EPOCHS = {
    "resnet50_imagenet": 90,
    "vitb16_imagenet": 90,
    "convnext_b_imagenet": 90,
    CLIP_PREFIX: 32,
}


def pct_of_training(prefix, epochs):
    """Epoch(s) -> percent of that architecture's full schedule."""
    total = TOTAL_EPOCHS[prefix]
    if np.isscalar(epochs):
        return 100.0 * epochs / (total - 1)
    return np.asarray([100.0 * e / (total - 1) for e in epochs], dtype=float)


def onset_epochs(prefix, dataset):
    """[epoch_early, epoch_mid, epoch_late] for an architecture, or None per slot.

    Read off the perturbed runs themselves rather than hard-coded: each run's
    first evaluated epoch IS its onset (the run is forked there).
    """
    out = []
    for onset in TIMINGS:
        ep = None
        for pert, _ in PERTS:
            pt = perturb_point(prefix, dataset, pert, onset)
            if pt is not None:
                ep = pt["epoch"]
                break
        out.append(ep)
    return out

# The 8 of 27 THINGS-fMRI ROIs that carry the perturbation effect, in legend
# order (roughly spared -> hit hardest). Lives here rather than in either figure
# script because BOTH the critical-window figure and the per-ROI trajectory
# figure plot exactly this subset, and they must agree: a reader comparing the
# two needs the same ROI set drawn in the same colors (both take colors from
# figS_baseline_neural_alignment.roi_style). Putting it in one of the two
# figure scripts would make the other import it and close an import cycle,
# since the highlighted figure already imports roi_style from the trajectory one.
#
#   va-1 (V1)                 earliest visual cortex -- consistently near zero
#   lRSC / rRSC               scene-selective -- also near zero
#   va-5..va-8 (VO1/VO2/LO1/LO2)  ventral/lateral occipital -- largest hits
#   LOC (bilateral)           object-selective -- hardest hit in every panel
THINGS_HIGHLIGHT_ROIS = [
    ("va-1", "V1 (visual)"),
    ("lRSC", "RSC (L, scene)"),
    ("rRSC", "RSC (R, scene)"),
    ("va-5", "VO1"),
    ("va-6", "VO2"),
    ("va-7", "LO1"),
    ("va-8", "LO2"),
    ("LOC", "LOC (object)"),
]


def _run_dir(prefix: str, run_suffix: str) -> Path:
    if run_suffix == "baseline":
        run = baseline_run(prefix)
    else:
        run = seeded(f"{prefix}_{run_suffix}" if run_suffix else prefix)
    return frrsa_dir(run)


def _frrsa_stem(prefix: str, dataset: str, corrected: bool) -> str:
    tag = f"_{frrsa_corrected_tag(prefix)}" if corrected else ""
    return f"inference_results_{dataset}_frrsa{tag}_epoch"


def _json_path(prefix: str, run_suffix: str, dataset: str, epoch: int,
               corrected: bool = False) -> Path:
    return _run_dir(prefix, run_suffix) / f"{_frrsa_stem(prefix, dataset, corrected)}{epoch}.json"


def _load_blocks(path: Path):
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _mean_sem(blocks: dict):
    """Mean and SEM of insample_score across ROIs in one result file."""
    scores = np.array([b["insample_score"] for b in blocks.values()
                       if b.get("insample_score") is not None], float)
    if scores.size == 0:
        return np.nan, np.nan
    sem = scores.std(ddof=1) / np.sqrt(scores.size) if scores.size > 1 else 0.0
    return float(scores.mean()), float(sem)


def _discover_epochs(prefix: str, run_suffix: str, dataset: str, corrected: bool = False):
    pat = str(_run_dir(prefix, run_suffix) / f"{_frrsa_stem(prefix, dataset, corrected)}*.json")
    eps = []
    for fp in glob.glob(pat):
        m = re.search(r"_epoch(\d+)\.json$", fp)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def baseline_trajectory(prefix: str, dataset: str, corrected: bool = False):
    """Return dict(epochs, mean, sem, per_roi) for an architecture's baseline run.

    per_roi is {roi: {epoch: score}} for downstream per-ROI plots.
    """
    epochs = _discover_epochs(prefix, "baseline", dataset, corrected)
    means, sems, per_roi = [], [], {}
    kept = []
    for e in epochs:
        blocks = _load_blocks(_json_path(prefix, "baseline", dataset, e, corrected))
        if blocks is None:
            continue
        m, s = _mean_sem(blocks)
        kept.append(e)
        means.append(m)
        sems.append(s)
        for roi, b in blocks.items():
            per_roi.setdefault(roi, {})[e] = b.get("insample_score")
    return {"epochs": np.array(kept), "mean": np.array(means),
            "sem": np.array(sems), "per_roi": per_roi}


def _point_at(prefix: str, dataset: str, run_suffix: str, epoch: int):
    blocks = _load_blocks(_json_path(prefix, run_suffix, dataset, epoch))
    if blocks is None:
        return None
    m, s = _mean_sem(blocks)
    per_roi = {roi: b.get("insample_score") for roi, b in blocks.items()}
    return {"epoch": epoch, "mean": m, "sem": s, "per_roi": per_roi}


def perturb_point(prefix: str, dataset: str, pert: str, onset: str):
    """(epoch, mean, sem, per_roi_scores) at the perturbation ONSET epoch, or None.

    The onset is the earliest evaluated epoch of a perturbed run: the run is forked
    from its baseline at the onset, so no earlier checkpoint exists.
    """
    epochs = _discover_epochs(prefix, f"{pert}_{onset}", dataset)
    if not epochs:
        return None
    return _point_at(prefix, dataset, f"{pert}_{onset}", epochs[0])


def perturb_point_final(prefix: str, dataset: str, pert: str, onset: str):
    """Same, at the LAST evaluated epoch of the run (end of training), or None.

    Returns None when only the onset epoch has been evaluated, so callers can
    distinguish "has not recovered" from "recovery was never measured".
    """
    epochs = _discover_epochs(prefix, f"{pert}_{onset}", dataset)
    if len(epochs) < 2:
        return None
    return _point_at(prefix, dataset, f"{pert}_{onset}", epochs[-1])


def baseline_at_epoch(prefix: str, dataset: str, epoch: int):
    """Baseline score at an EXACT epoch (not nearest), or None if not evaluated."""
    return _point_at(prefix, dataset, "baseline", epoch)


# The baseline sweep evaluates every 5th epoch, but three perturbation onsets
# do not land on that grid (ViT-B/16 mid = 11, ConvNeXt-B early = 3 and mid = 6).
# Substituting the nearest evaluated baseline epoch there compares a perturbed
# model at epoch E against a baseline at epoch E+-2, which is not a controlled
# comparison while the baseline is still climbing steeply -- and ConvNeXt-B's
# early AND mid onsets both fall back to the SAME baseline epoch (5), so its
# early-vs-mid contrast currently has no baseline contrast in it at all.
#
# STRICT (default) makes that fail loudly instead of silently. Set
# TP_FRRSA_ALLOW_NEAREST=1 to restore the old behavior for a quick preview.
STRICT_EPOCH_MATCH = os.environ.get("TP_FRRSA_ALLOW_NEAREST", "") not in ("1", "true", "yes")

_FILL_CMD = (
    "python pipeline/scripts/evaluation/run_frrsa.py"
)


def baseline_at_nearest_epoch(traj: dict, epoch: int):
    """Baseline mean (and the epoch it came from) matched to `epoch`.

    Exact match is required unless STRICT_EPOCH_MATCH is off, in which case the
    nearest evaluated epoch is substituted and a warning is printed.
    """
    eps = traj["epochs"]
    if len(eps) == 0:
        return np.nan, None
    j = int(np.argmin(np.abs(eps - epoch)))
    used = int(eps[j])
    if used != epoch:
        msg = (f"baseline has no FR-RSA evaluation at epoch {epoch} "
               f"(nearest is {used}); the comparison would be epoch-mismatched. "
               f"Run: {_FILL_CMD}")
        if STRICT_EPOCH_MATCH:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
    return float(traj["mean"][j]), used


def unmatched_onsets(onset_table: dict):
    """[(prefix, dataset, onset_epoch)] for every onset with no baseline eval.

    `onset_table` is {prefix: {timing_label: epoch}}. Use this to check coverage
    before regenerating figures, without having to trip the exception.
    """
    missing = []
    for prefix, onsets in onset_table.items():
        for dataset, _ in DATASETS:
            have = set(_discover_epochs(prefix, "baseline", dataset))
            for epoch in sorted(set(onsets.values())):
                if epoch not in have:
                    missing.append((prefix, dataset, epoch))
    return missing


# The baseline sweep's regular grid: every 5th epoch (CLIP-ViT-B/32, on its
# 32-epoch schedule, every 2nd) plus the final one. Onsets are evaluated on top.
BASELINE_EPOCH_STEP = {CLIP_PREFIX: 2}


def baseline_grid(prefix: str):
    """Epochs of an architecture's regular baseline grid."""
    total = TOTAL_EPOCHS[prefix]
    return sorted(set(range(0, total, BASELINE_EPOCH_STEP.get(prefix, 5))) | {total - 1})


def missing_baseline_epochs(prefix: str, dataset: str, corrected: bool = False):
    """Epochs of the regular baseline grid that have no FR-RSA result file."""
    return sorted(set(baseline_grid(prefix))
                  - set(_discover_epochs(prefix, "baseline", dataset, corrected)))


def fig_text_wrapped(fig, x, y, text, fontsize, avg_char_width_frac=0.52, **kwargs):
    """fig.text() that wraps to the ACTUAL available width instead of drawing one
    long unwrapped line.

    A `fig.text(..., ha='center')` call with a string longer than fits at its
    fontsize does not clip or wrap on its own -- it just draws past both figure
    edges. `get_tightbbox()` (used by the default `savefig.bbox: 'tight'`) then
    expands the SAVED canvas to include that overflow, silently inflating the
    exported figure past its intended `figsize` (e.g. a 7.5in-wide figure was
    exporting at 13.5in because one caption line ran ~12.6in wide at its font
    size). Wrapping first keeps the export at the intended size.

    Width available is estimated from the plotted subplot span (fig.subplotpars
    left/right), not the full canvas, since these captions are meant to sit over
    the plot area, not spill into a side legend's reserved margin.
    """
    import textwrap
    fig_w_in = fig.get_size_inches()[0]
    avail_in = (fig.subplotpars.right - fig.subplotpars.left) * fig_w_in
    char_w_in = fontsize / 72.0 * avg_char_width_frac
    max_chars = max(20, int(avail_in / char_w_in))
    wrapped = textwrap.fill(text, max_chars)
    return fig.text(x, y, wrapped, fontsize=fontsize, **kwargs)
