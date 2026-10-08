"""Supplementary figure: CLIP-HBA's 66 SPoSE output dimensions under one-epoch perturbations.

CLIP-HBA is CLIP ViT-L/14 fine-tuned with DoRA adapters to predict the 66
SPoSE human-similarity dimensions. Definitions live in the paper's figure
caption. Drawn at the paper's text width (5.5 in) with the shared `paper_style` grammar.

A  One THINGS image's 66-d output from the unperturbed model (top 6 dims).
B  Change in behavioral alignment (perturbed - baseline Spearman rho) at
   each of the 28 onsets, for all four perturbation types.
C  Random target: per-dimension change in response variance across the 48
   images, at the perturbed epoch and 10 epochs later.
D  Group means of C for onsets 3 and 89, semantic vs perceptual dimensions.

Reads the 48x66 output matrices written by
evaluation/extract_cliphba_things_embeddings.py:
  <EMB_ROOT>/baseline_seed{S}/<run>/eval_results/things_embeddings_epoch{E}.npy
  <EMB_ROOT>/{perturb}_perturb_seed{S}/epoch{E}_length1/<run>/eval_results/...
with EMB_ROOT = <results_root>/cliphba_embeddings, plus
pipeline/data/RDM48_triplet.mat (human triplet-judgment RDM).

Writes Fig4_spose_dimensions_defs_in_caption.{png,pdf} to <figures_root>.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import scipy.io as sio

VIZ_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIZ_DIR.parents[2]
sys.path.insert(0, str(VIZ_DIR))
import paper_style as S  # noqa: E402
from arch_runs import FIGURES as OUTDIR, RESULTS  # noqa: E402

S.apply()

SEMANTIC_DIMS = {
    1,   # food-related
    2,   # animal-related
    4,   # plant-related
    5,   # house-related; furnishing-related
    6,   # valuable; precious
    7,   # transportation; movement-related
    8,   # body; people-related
    10,  # electronics; technology
    12,  # outdoors
    15,  # hobby-related; game-related; playing-related
    17,  # fluid-related; drink-related
    18,  # water-related
    25,  # weapon-related; war-related; dangerous
    27,  # household-related
    28,  # feminine
    29,  # body-part-related
    31,  # music-related; hearing-related; hobby-related; loud
    34,  # construction-related; craftsmanship-related; housework-related
    37,  # seating; standing; lying-related
    38,  # flying-related; sky-related
    39,  # bug-related; non-mammalian; disgusting
    43,  # bathroom-related; wetness-related
    45,  # heat-related; fire-related; light-related
    47,  # foot-related; walking-related
    50,  # head-related
    53,  # child-related; toy-related; cute
    54,  # farm-related; historical
    55,  # seeing-related
    56,  # medicine-related; health-related
    57,  # sweet; dessert-related
    61,  # coldness-related; winter-related
    62,  # measurement-related; numbers-related
    64,  # masculine
}

PERCEPTUAL_DIMS = {
    0,   # metallic; artificial
    3,   # textile
    9,   # wood-related; brown
    11,  # colorful; playful
    13,  # circular; round
    14,  # paper-related; flat
    16,  # tools-related; handheld; elongated
    19,  # oriented; many; plenty
    20,  # powdery; earth-related; waste-related
    21,  # white
    22,  # coarse-scale pattern; many things
    23,  # red
    24,  # long; thin
    26,  # black
    30,  # tubular
    32,  # grid-related; grating-related
    33,  # repetitive; spiky
    35,  # spherical; voluminous
    36,  # string-related; stringy; curved
    40,  # transparent; shiny; crystalline
    41,  # sand-colored
    42,  # green
    44,  # yellow
    46,  # beams-related; mesh-related
    48,  # box-related; container
    49,  # stick-shaped; container
    51,  # upright; elongated; volumous
    52,  # pointed; spiky
    58,  # orange
    59,  # thin; flat; wrapping
    60,  # cylindrical; conical; cushioning
    63,  # fluffy; soft
    65,  # fine-grained; pattern
}

assert len(SEMANTIC_DIMS) + len(PERCEPTUAL_DIMS) == 66
assert SEMANTIC_DIMS.isdisjoint(PERCEPTUAL_DIMS)

MAX_EPOCH = 117


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perturb", type=str, default="random_target",
                   help="Perturbation type to focus on (default: random_target)")
    p.add_argument("--offset", type=int, default=10,
                   help="Offset value for the recovery heatmap (default: 10)")
    return p.parse_args()


ALL_PERTURBATIONS = ["random_target", "label_shuffle", "image_noise", "uniform_images"]

EMB_ROOT = RESULTS / "cliphba_embeddings"
HUMAN_RDM_PATH = REPO_ROOT / "pipeline" / "data" / "RDM48_triplet.mat"
SEEDS = [1, 2, 3]


def _upper_tri(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices_from(mat, k=1)]


def _run_dir(el_dir: Path, pt: str) -> Path | None:
    """The one run folder for (perturbation, onset, seed), or None.

    Backup copies (e.g. `..._init-seed1behavioral-rsa-True__lr1e-4_backup`)
    sit beside the real run and sort ahead of it, so an unfiltered glob would
    read the backup.
    """
    runs = [r for r in el_dir.glob(f"vit_l_14_rank32_perturb-type-{pt}_*")
            if "backup" not in r.name]
    if len(runs) > 1:
        raise ValueError(f"{el_dir}: {len(runs)} non-backup run folders: "
                         f"{[r.name for r in runs]}")
    return runs[0] if runs else None


def _full_behavioral_rsa(emb: np.ndarray, human_upper: np.ndarray) -> float:
    """Full 66-d Pearson RDM vs human RDM, Spearman rho."""
    rdm = 1.0 - np.corrcoef(emb)
    np.fill_diagonal(rdm, 0)
    rho, _ = spearmanr(_upper_tri(rdm), human_upper)
    return rho if not np.isnan(rho) else 0.0


def _compute_full_behavioral_delta_rho() -> dict[str, pd.DataFrame]:
    """Load embeddings directly and compute full behavioral RSA delta per onset.

    Returns {perturb_type: DataFrame with columns [epoch, seed, rho_pert, rho_base, delta_rho]}.
    """
    import re

    human_rdm = sio.loadmat(HUMAN_RDM_PATH)["RDM48_triplet"].astype(np.float64)
    human_upper = _upper_tri(human_rdm)

    def _emb_path(run_dir, epoch):
        p = run_dir / "eval_results" / f"things_embeddings_epoch{epoch}.npy"
        return p if p.exists() else None

    baselines = {}
    for seed in SEEDS:
        run = (EMB_ROOT / f"baseline_seed{seed}" /
               f"vit_l_14_rank32_perturb-type-none_init-seed{seed}behavioral-rsa-True")
        for npy in (run / "eval_results").glob("things_embeddings_epoch*.npy"):
            ep = int(re.search(r"epoch(\d+)", npy.name).group(1))
            baselines[(seed, ep)] = np.load(npy)

    results = {}
    for pt in ALL_PERTURBATIONS:
        rows = []
        for seed in SEEDS:
            seed_dir = EMB_ROOT / f"{pt}_perturb_seed{seed}"
            if not seed_dir.exists():
                continue
            for el_dir in sorted(seed_dir.glob("epoch*_length1")):
                m = re.match(r"epoch(\d+)_length1$", el_dir.name)
                if not m:
                    continue
                onset = int(m.group(1))
                run = _run_dir(el_dir, pt)
                if run is None:
                    continue
                p = _emb_path(run, onset)
                if p is None:
                    continue
                emb_pert = np.load(p)
                emb_base = baselines.get((seed, onset))
                if emb_base is None:
                    continue
                rho_p = _full_behavioral_rsa(emb_pert, human_upper)
                rho_b = _full_behavioral_rsa(emb_base, human_upper)
                rows.append({"epoch": onset, "seed": seed,
                             "rho_pert": rho_p, "rho_base": rho_b,
                             "delta_rho": rho_p - rho_b})
        if rows:
            results[pt] = pd.DataFrame(rows)
    return results


def _load_dim_labels() -> list[str]:
    ns = {}
    label_path = REPO_ROOT / "temporal_perturbations" / "data" / "spose_dimensions.py"
    exec(label_path.read_text(), ns)
    return ns["classnames66"]


def _load_embeddings(perturb_types: list[str] | None = None, offset: int = 0):
    """Load raw (48,66) embeddings. Returns (baselines, perturbations) dicts.

    baselines:     {(seed, epoch): (48,66) ndarray}
    perturbations: {(perturb_type, seed, onset_epoch): (48,66) ndarray}
    """
    import re

    perturb_types = perturb_types or ALL_PERTURBATIONS

    baselines = {}
    for seed in SEEDS:
        run = (EMB_ROOT / f"baseline_seed{seed}" /
               f"vit_l_14_rank32_perturb-type-none_init-seed{seed}behavioral-rsa-True")
        for npy in (run / "eval_results").glob("things_embeddings_epoch*.npy"):
            ep = int(re.search(r"epoch(\d+)", npy.name).group(1))
            baselines[(seed, ep)] = np.load(npy)

    perturbations = {}
    for pt in perturb_types:
        for seed in SEEDS:
            seed_dir = EMB_ROOT / f"{pt}_perturb_seed{seed}"
            if not seed_dir.exists():
                continue
            for el_dir in sorted(seed_dir.glob("epoch*_length1")):
                m = re.match(r"epoch(\d+)_length1$", el_dir.name)
                if not m:
                    continue
                onset = int(m.group(1))
                eval_epoch = onset + offset
                run = _run_dir(el_dir, pt)
                if run is None:
                    continue
                p = run / "eval_results" / f"things_embeddings_epoch{eval_epoch}.npy"
                if not p.exists():
                    continue
                perturbations[(pt, seed, onset)] = np.load(p)

    return baselines, perturbations


def _compute_delta_variance(baselines, perturbations, perturb: str,
                            offset: int = 0) -> pd.DataFrame:
    """Per-dimension fractional Δ variance: (var_pert - var_base) / var_base."""
    dim_labels = _load_dim_labels()
    rows = []
    for (pt, seed, onset), emb_p in perturbations.items():
        if pt != perturb:
            continue
        eval_epoch = onset + offset
        emb_b = baselines.get((seed, eval_epoch if offset == 0 else eval_epoch))
        if emb_b is None:
            emb_b = baselines.get((seed, onset))
        if emb_b is None:
            continue
        var_p = np.var(emb_p, axis=0)
        var_b = np.var(emb_b, axis=0)
        for d in range(66):
            denom = var_b[d] if var_b[d] > 1e-12 else 1e-12
            rows.append({
                "epoch": onset,
                "seed": seed,
                "dim_idx": d,
                "dim_label": dim_labels[d],
                "delta_var": (var_p[d] - var_b[d]) / denom,
            })
    return pd.DataFrame(rows).groupby(
        ["epoch", "dim_idx", "dim_label"]
    )["delta_var"].mean().reset_index()


def _build_heatmap_matrix(delta_df: pd.DataFrame, dim_order: np.ndarray,
                          value_col: str = "delta_var"):
    epochs = sorted(delta_df["epoch"].unique())
    n_dims = len(dim_order)
    mat = np.full((n_dims, len(epochs)), np.nan)
    for j, ep in enumerate(epochs):
        ep_data = delta_df[delta_df["epoch"] == ep].set_index("dim_idx")[value_col]
        for i, dim_idx in enumerate(dim_order):
            if dim_idx in ep_data.index:
                mat[i, j] = ep_data[dim_idx]
    return mat, epochs


def _dim_sort_order(baselines: dict, all_dims: set | None = None) -> np.ndarray:
    """Semantic dims on top, perceptual on bottom; each group sorted by
    mean baseline variance (highest first)."""
    stacked = np.stack(list(baselines.values()), axis=0)  # (N, 48, 66)
    per_dim_var = np.var(stacked.reshape(-1, 66), axis=0)  # (66,)

    dims = all_dims or set(range(66))
    sem = sorted([d for d in dims if d in SEMANTIC_DIMS],
                 key=lambda d: per_dim_var[d], reverse=True)
    perc = sorted([d for d in dims if d in PERCEPTUAL_DIMS],
                  key=lambda d: per_dim_var[d], reverse=True)
    return np.array(sem + perc)


# Drawing (print size, paper_style grammar)

FIG_H = 5.51

# Panel A shows the unperturbed model's output for one of the 48 THINGS
# images. Row order of every things_embeddings_epoch{E}.npy written by
# inference_core is recorded in the matching things_concepts_epoch{E}.json,
# which is what makes the image identification unambiguous.
EXAMPLE_IMAGE = "tuba_01b.jpg"
EXAMPLE_SEED = 1
EXAMPLE_EPOCH = 99
N_EXAMPLE_DIMS = 6

EARLY_ONSET = 3
LATE_ONSET = 89

ROW_LABEL_STEP = 5
N_ROW_LABELS_PER_GROUP = 6

# Onset colour is reserved for onset timing, so the four perturbation types
# are two greys; line style + marker carry the type (paper_style grammar).
PERT_GREY = {
    "random_target":  "#2e2e2e",
    "label_shuffle":  "#8c8c8c",
    "image_noise":    "#2e2e2e",
    "uniform_images": "#8c8c8c",
}
SEMANTIC_SHADE = "#4d4d4d"
PERCEPTUAL_SHADE = "#c2c2c2"
HEATMAP_CMAP = "PiYG"
# Group labels in C and D carry A's bar shades, so the semantic/perceptual
# split reads the same in all three panels.
GROUP_TEXT_STYLE = {
    "semantic": dict(color="white", bbox=dict(boxstyle="round,pad=0.22",
                                              facecolor=SEMANTIC_SHADE, edgecolor="none")),
    "perceptual": dict(color="black", bbox=dict(boxstyle="round,pad=0.22",
                                                facecolor=PERCEPTUAL_SHADE, edgecolor="none")),
}
MINUS = "−"


def _fmt(v: float, nd: int = 2) -> str:
    s = f"{v:+.{nd}f}" if v > 0 else f"{v:.{nd}f}"
    return s.replace("-", MINUS)


def _tick_fmt(v, _pos=None):
    return "0" if abs(v) < 1e-9 else _fmt(v, 1)


def _onset_to_step(x):
    """Onset epoch -> position on the onset grid (0-9 every epoch, then every 5)."""
    x = np.asarray(x, dtype=float)
    return np.where(x <= 9, x, 9 + (x - 9) / 5)


def _step_to_onset(u):
    u = np.asarray(u, dtype=float)
    return np.where(u <= 9, u, 9 + (u - 9) * 5)


def _first_name(label: str) -> str:
    return label.split(";")[0].strip()


def _add_axes_in(fig, x0, top, w, h):
    """Axes placed in inches, `top` measured down from the top edge."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x0 / W, (H - top - h) / H, w / W, h / H])


def _load_example_output():
    run = (EMB_ROOT / f"baseline_seed{EXAMPLE_SEED}" /
           f"vit_l_14_rank32_perturb-type-none_init-seed{EXAMPLE_SEED}behavioral-rsa-True"
           / "eval_results")
    names = json.loads((run / f"things_concepts_epoch{EXAMPLE_EPOCH}.json").read_text())
    emb = np.load(run / f"things_embeddings_epoch{EXAMPLE_EPOCH}.npy")
    if len(names) != emb.shape[0] or EXAMPLE_IMAGE not in names:
        raise ValueError(f"{EXAMPLE_IMAGE} not a row of {run}: {len(names)} names, "
                         f"matrix {emb.shape}")
    return emb[names.index(EXAMPLE_IMAGE)]


def panel_a_example(ax, values: np.ndarray, dim_labels: list[str]):
    top = np.argsort(values)[::-1][:N_EXAMPLE_DIMS]
    y = np.arange(len(top))
    shades = [SEMANTIC_SHADE if d in SEMANTIC_DIMS else PERCEPTUAL_SHADE for d in top]
    ax.barh(y, values[top], height=0.68, color=shades, edgecolor="#333333",
            linewidth=0.4)
    ax.set_yticks(np.arange(len(top) + 1))
    ax.set_yticklabels([_first_name(dim_labels[d]) for d in top] +
                       [f"+ {66 - N_EXAMPLE_DIMS} more"])
    ax.get_yticklabels()[-1].set_color("#333333")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(top) + 0.5, -0.6)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 25, 50])
    ax.set_xlabel("Predicted value\n(SPoSE targets span 0–100)", linespacing=1.15)
    # Centred over bars + their labels, so the two lines stay clear of B.
    ax.set_title("THINGS image “tuba”", fontsize=S.FS_LABEL, pad=12, x=0.12)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(facecolor=SEMANTIC_SHADE, edgecolor="#333333",
                             linewidth=0.4, label="semantic"),
                       Patch(facecolor=PERCEPTUAL_SHADE, edgecolor="#333333",
                             linewidth=0.4, label="perceptual")],
              loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=S.FS_NOTE,
              handlelength=0.9, handleheight=0.7, handletextpad=0.35,
              labelspacing=0.25, borderaxespad=0.0)


def _onset_stars(ax, xs, y_axes, size=S.MS_STAR):
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, timing in zip(xs, ("early", "late")):
        ax.plot([x], [y_axes], marker="*", markersize=size, markeredgewidth=0,
                color=S.ONSET_COLORS[timing], transform=tr, clip_on=False,
                zorder=15, linestyle="none")


def panel_b_alignment(ax, behavioral_deltas: dict[str, pd.DataFrame]):
    """Full 66-d behavioral RSA delta (perturbed - baseline) at each onset."""
    ax.axhline(0, color=S.BASELINE_COLOR, linewidth=S.LW_BASELINE, zorder=1)
    ends = {}
    means = {}
    for pt in ["uniform_images", "image_noise", "label_shuffle", "random_target"]:
        if pt not in behavioral_deltas:
            continue
        df = behavioral_deltas[pt]
        by_epoch = df.groupby("epoch")["delta_rho"].agg(
            mean="mean",
            sem="sem",
        ).reset_index().sort_values("epoch")
        color = PERT_GREY[pt]
        ax.fill_between(by_epoch["epoch"],
                        by_epoch["mean"] - by_epoch["sem"],
                        by_epoch["mean"] + by_epoch["sem"],
                        color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(by_epoch["epoch"], by_epoch["mean"], color=color,
                linewidth=S.LW_DATA * 0.85, linestyle=S.pert_linestyle(pt), zorder=3,
                **S.marker_kw(pt, color, ms=S.MS * 0.72, mew=0.7))
        ends[pt] = by_epoch["mean"].iloc[-1]
        means[pt] = by_epoch.set_index("epoch")["mean"]

    # Same onset spacing as C's columns: one step per onset, so the ten
    # one-epoch-apart early onsets are as legible as the 5-epoch grid.
    ax.set_xscale("function", functions=(_onset_to_step, _step_to_onset))
    ax.set_xlim(-0.8, _step_to_onset(27.8))
    ax.set_ylim(-0.30, 0.10)
    ax.set_xticks([0, 3, 6, 9, 29, 49, 69, 99])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    onsets = sorted(set().union(*(m.index for m in means.values())))
    ax.set_xticks(onsets, minor=True)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.2, width=0.4)
    ax.set_yticks([-0.3, -0.2, -0.1, 0.0, 0.1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_tick_fmt))
    ax.set_xlabel("Onset of the perturbed epoch")
    ax.set_ylabel("Δ behavioral alignment\n(Spearman ρ),\nperturbed − baseline",
                  linespacing=1.1)

    # Direct labels at the line ends, pushed apart where two ends nearly
    # coincide (the two input-side lines end 0.02 apart).
    fig = ax.figure
    ax_h_in = ax.get_position().height * fig.get_figheight()
    y0, y1 = ax.get_ylim()
    min_sep = (S.FS_LEGEND * 1.2 / 72.0) / ax_h_in * (y1 - y0)
    placed = []
    for pt, y in sorted(ends.items(), key=lambda kv: kv[1]):
        if placed and y - placed[-1][1] < min_sep:
            y = placed[-1][1] + min_sep
        placed.append((pt, y))
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for pt, y in placed:
        ax.text(1.015, y, S.PERT_LABEL[pt], transform=tr, ha="left", va="center",
                fontsize=S.FS_LEGEND, color=PERT_GREY[pt], clip_on=False)

    _onset_stars(ax, [EARLY_ONSET, LATE_ONSET], 0.035)


def panel_c_heatmaps(ax_left, ax_right, delta_onset, delta_offset, dim_order,
                     offset_val):
    mat_onset, epochs_onset = _build_heatmap_matrix(delta_onset, dim_order, "delta_var")
    mat_offset, epochs_offset = _build_heatmap_matrix(delta_offset, dim_order, "delta_var")

    # Symmetric limits: equal colour intensity means equal |ΔVar| on both
    # sides of zero; ΔVar cannot fall below -1 (variance is non-negative).
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    dim_labels = delta_onset.drop_duplicates("dim_idx").set_index("dim_idx")["dim_label"]
    n_sem = sum(1 for d in dim_order if d in SEMANTIC_DIMS)
    n_dims = len(dim_order)

    labeled_rows = ([i * ROW_LABEL_STEP for i in range(N_ROW_LABELS_PER_GROUP)] +
                    [n_sem + i * ROW_LABEL_STEP for i in range(N_ROW_LABELS_PER_GROUP)])

    greyed = set()
    for j, ep in enumerate(epochs_offset):
        onset_ep = ep - offset_val
        if onset_ep + offset_val > MAX_EPOCH:
            mat_offset[:, j] = np.nan
            greyed.add(j)

    tick_epochs = [0, 9, 29, 59, 99]
    for ax, mat, epochs in ((ax_left, mat_onset, epochs_onset),
                            (ax_right, mat_offset, epochs_offset)):
        im = ax.pcolormesh(np.arange(len(epochs) + 1), np.arange(n_dims + 1), mat,
                           cmap=HEATMAP_CMAP, norm=norm, edgecolors="face",
                           linewidth=0, rasterized=True)
        ax.set_xlim(0, len(epochs))
        ax.set_ylim(n_dims, 0)
        ax.set_xticks([epochs.index(e) + 0.5 for e in tick_epochs])
        ax.set_xticklabels([str(e) for e in tick_epochs])
        ax.set_xticks(np.arange(len(epochs)) + 0.5, minor=True)
        ax.tick_params(axis="x", which="minor", length=1.2, width=0.4)
        ax.axhline(n_sem, color="white", linewidth=1.4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        _onset_stars(ax, [epochs.index(EARLY_ONSET) + 0.5, epochs.index(LATE_ONSET) + 0.5],
                     1.03, size=S.MS_STAR * 0.9)

    ax_left.set_yticks([i + 0.5 for i in labeled_rows])
    ax_left.set_yticklabels([_first_name(dim_labels[dim_order[i]]) for i in labeled_rows])
    ax_left.tick_params(axis="y", length=2)
    ax_right.set_yticks([])

    for col_j in greyed:
        ax_right.axvspan(col_j, col_j + 1, color="#DDDDDD", zorder=2)
        ax_right.text(col_j + 0.5, n_dims / 2, f"no checkpoint {offset_val} epochs later",
                      ha="center", va="center", fontsize=S.FS_NOTE, color=S.NOTE_COLOR,
                      rotation=90, zorder=3)

    for ax, title in ((ax_left, "measured at the\nperturbed epoch"),
                      (ax_right, f"measured {offset_val} epochs after\nthe perturbed epoch")):
        ax.set_title(title, fontsize=S.FS_LABEL, pad=11.6, linespacing=1.1)
    return im, n_sem, n_dims


def panel_d_group_means(ax_sem, ax_perc, delta_onset, delta_offset, offset_val):
    def _get_means(delta_df, dim_set, epoch):
        sub = delta_df[(delta_df["epoch"] == epoch) & (delta_df["dim_idx"].isin(dim_set))]
        return sub["delta_var"].mean(), sub["delta_var"].sem()

    for ax, name, dim_set in ((ax_sem, "semantic", SEMANTIC_DIMS),
                              (ax_perc, "perceptual", PERCEPTUAL_DIMS)):
        ax.axhline(0, color=S.BASELINE_COLOR, linewidth=S.LW_BASELINE, zorder=1)
        summary = []
        for onset_ep, timing in ((LATE_ONSET, "late"), (EARLY_ONSET, "early")):
            m0, s0 = _get_means(delta_onset, dim_set, onset_ep)
            m1, s1 = _get_means(delta_offset, dim_set, onset_ep)
            c = S.ONSET_COLORS[timing]
            ax.plot([0, offset_val], [m0, m1], color=c, linewidth=S.LW_DATA, zorder=3)
            ax.errorbar([0, offset_val], [m0, m1], yerr=[s0, s1], fmt="none",
                        ecolor=c, elinewidth=0.8, capsize=0, zorder=4)
            ax.plot([0], [m0], marker="*", markersize=S.MS_STAR, markeredgewidth=0,
                    color=c, linestyle="none", zorder=5)
            ax.plot([offset_val], [m1], marker="o", markersize=S.MS * 0.9,
                    markeredgewidth=0, color=c, linestyle="none", zorder=5)
            summary.append((f"{_fmt(m0)} → {_fmt(m1)}", c))
        for k, (txt, c) in enumerate(summary):
            ax.text(1.0, 0.07 + 0.15 * k, txt, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=S.FS_NOTE, color=c)
        ax.set_xlim(-2.0, offset_val + 2.5)
        ax.set_ylim(-1.06, 0.09)
        ax.set_yticks([-1.0, -0.5, 0.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_tick_fmt))
        ax.set_xticks([0, offset_val])
        ax.set_title(name, fontsize=S.FS_LABEL, pad=5,
                     **GROUP_TEXT_STYLE[name])
    ax_sem.set_xticklabels([])
    ax_perc.set_xticklabels(["0", f"+{offset_val}"])
    ax_perc.set_xlabel("Epochs after the\nperturbed epoch", linespacing=1.1)


def _top_legend(fig):
    """Two centred rows: onset colours + baseline, then the input/target-side grammar."""
    grey = "#555555"
    rows = [
        [Line2D([], [], color=S.ONSET_COLORS["early"], lw=S.LW_DATA, marker="*",
                markersize=S.MS_STAR, markeredgewidth=0,
                label=f"early onset (epoch {EARLY_ONSET})"),
         Line2D([], [], color=S.ONSET_COLORS["late"], lw=S.LW_DATA, marker="*",
                markersize=S.MS_STAR, markeredgewidth=0,
                label=f"late onset (epoch {LATE_ONSET})"),
         Line2D([], [], color=S.BASELINE_COLOR, lw=S.LW_BASELINE,
                label="unperturbed baseline")],
        [Line2D([], [], color=grey, lw=S.LW_DATA, ls="-",
                label="input-side perturbation (filled marker, solid line)",
                **S.marker_kw("image_noise", grey)),
         Line2D([], [], color=grey, lw=S.LW_DATA, ls=(0, (3.5, 1.8)),
                label="target-side perturbation (open marker, dashed line)",
                **S.marker_kw("label_shuffle", grey))],
    ]
    H = fig.get_figheight()
    for k, handles in enumerate(rows):
        leg = fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                         bbox_to_anchor=(0.5, 1 - (0.04 + 0.16 * k) / H),
                         fontsize=S.FS_LEGEND, handlelength=2.2, columnspacing=1.4,
                         handletextpad=0.45, borderaxespad=0.0)
        fig.add_artist(leg)


def make_figure():
    args = _parse_args()
    perturb = args.perturb
    offset_val = args.offset

    print("Loading embeddings...")
    baselines_onset, perts_onset = _load_embeddings([perturb], offset=0)
    print(f"  Onset: {len(baselines_onset)} baselines, {len(perts_onset)} perturbations")

    delta_onset = _compute_delta_variance(baselines_onset, perts_onset, perturb, offset=0)
    print(f"  Onset delta_var: {len(delta_onset)} rows, "
          f"{delta_onset['epoch'].nunique()} epochs, "
          f"{delta_onset['dim_idx'].nunique()} dims")

    baselines_offset, perts_offset = _load_embeddings([perturb], offset=offset_val)
    if not perts_offset:
        raise FileNotFoundError(
            f"No onset+{offset_val} embeddings for {perturb}; run "
            f"pipeline/scripts/analysis/generate_dim_embeddings.py --offset {offset_val}")
    delta_offset = _compute_delta_variance(
        baselines_onset, perts_offset, perturb, offset=offset_val)
    print(f"  Offset delta_var: {len(delta_offset)} rows, "
          f"{delta_offset['epoch'].nunique()} epochs")

    print("Computing full behavioral RSA for Panel B...")
    behavioral_deltas = _compute_full_behavioral_delta_rho()
    print(f"Panel B: {len(behavioral_deltas)} perturbation types, "
          f"full 66-d Pearson RDM vs human RDM")

    dim_order = _dim_sort_order(baselines_onset)
    dim_labels = _load_dim_labels()
    example = _load_example_output()

    fig = plt.figure(figsize=(S.TEXT_WIDTH, FIG_H))
    W, H = fig.get_size_inches()

    _top_legend(fig)

    # --- row 1: A | B -------------------------------------------------------
    row1_top = 0.64
    ax_a = _add_axes_in(fig, 0.80, row1_top + 0.26, 0.92, 1.19)
    ax_b = _add_axes_in(fig, 2.66, row1_top, 2.06, 1.45)
    panel_a_example(ax_a, example, dim_labels)
    panel_b_alignment(ax_b, behavioral_deltas)

    # --- row 2: C | D -------------------------------------------------------
    hm_top, hm_h, hm_w, hm_gap = row1_top + 2.47, 1.95, 1.15, 0.14
    hm_left = 1.04
    ax_cl = _add_axes_in(fig, hm_left, hm_top, hm_w, hm_h)
    ax_cr = _add_axes_in(fig, hm_left + hm_w + hm_gap, hm_top, hm_w, hm_h)
    hm_right = hm_left + 2 * hm_w + hm_gap
    cax = _add_axes_in(fig, hm_right + 0.07, hm_top + 0.2, 0.07, hm_h - 0.4)
    im, n_sem, n_dims = panel_c_heatmaps(ax_cl, ax_cr, delta_onset, delta_offset,
                                         dim_order, offset_val)
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels([f"{MINUS}1", f"{MINUS}0.5", "0", "+0.5", "+1"])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2, width=0.5)
    cb.set_label("ΔVar (fractional change)", fontsize=S.FS_LABEL, labelpad=3)

    for y_mid, text in ((n_sem / 2, f"semantic ({n_sem})"),
                        (n_sem + (n_dims - n_sem) / 2, f"perceptual ({n_dims - n_sem})")):
        y_in = hm_top + hm_h * y_mid / n_dims
        fig.text(0.10 / W, 1 - y_in / H, text, rotation=90, ha="center", va="center",
                 fontsize=S.FS_LABEL, **GROUP_TEXT_STYLE[text.split(" ")[0]])

    d_left, d_w, d_gap = 4.46, 0.98, 0.34
    d_h = (hm_h - d_gap) / 2
    ax_ds = _add_axes_in(fig, d_left, hm_top, d_w, d_h)
    ax_dp = _add_axes_in(fig, d_left, hm_top + d_h + d_gap, d_w, d_h)
    panel_d_group_means(ax_ds, ax_dp, delta_onset, delta_offset, offset_val)
    fig.text((d_left - 0.33) / W, 1 - (hm_top + hm_h / 2) / H,
             "Mean ΔVar over 33 dimensions", rotation=90, ha="center",
             va="center", fontsize=S.FS_LABEL)

    fig.text((hm_left + hm_right) / 2 / W, 1 - (hm_top + hm_h + 0.19) / H,
             "Onset of the perturbed epoch",
             ha="center", va="top", fontsize=S.FS_LABEL)

    S.panel_header(fig, [ax_a, ax_b], "A", "Model output for one image",
                   x_axes=ax_a)
    S.panel_header(fig, [ax_b, ax_a], "B", "Δ behavioral alignment vs. onset",
                   x_axes=ax_b)
    S.panel_header(fig, [ax_cl, ax_cr, ax_ds], "C",
                   "Random target: ΔVar per SPoSE dimension at 28 onsets",
                   x_axes=ax_a)
    S.panel_header(fig, [ax_ds, ax_cl, ax_cr], "D", "Group-mean ΔVar", x_axes=ax_ds)

    S.save(fig, OUTDIR, "Fig4_spose_dimensions_defs_in_caption")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
