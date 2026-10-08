"""TriMap 2D projection of cross-architecture penultimate features: baseline vs perturbed.

Loads runs_root checkpoints (ViT-B/16, ConvNeXt-B, RN50) via the Lightning
checkpoint path, extracts penultimate features on ImageNet val, runs TriMap
independently on each condition, and produces side-by-side scatter plots
colored by Muttenthaler et al. 2025 superordinate categories.

TriMap coordinates go to `<results_root>/trimap/`, the scatter plots to
`<figures_root>/`. Embeddings are cached (features in
`<feature_cache>/cross_arch_embeddings/`, image synset order in
`<results_root>/imagenet_embeddings/`).

Usage
-----
    python pipeline/scripts/visualization/trimap_imagenet.py   # the four Fig 3 maps
    python pipeline/scripts/visualization/trimap_imagenet.py \\
        --architecture "ViT-B/16" --run-baseline vitb16_imagenet_baseline \\
        --run-perturbed vitb16_imagenet_image_noise_late --epoch 70
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

VIZ_DIR = Path(__file__).resolve().parent
PIPELINE = VIZ_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths

CACHE_DIR = paths.path("feature_cache") / "cross_arch_embeddings"
SYNSETS_DIR = paths.results_root() / "imagenet_embeddings"
COORDS_DIR = paths.results_root() / "trimap"
FIGURES_DIR = paths.path("figures_root")

# The four maps Fig 3 (fig3_storyboard.py) reads, drawn when no run is named.
SEED_SUFFIX = os.environ.get("TP_SEED_SUFFIX", "")
PAPER_EPOCH = 70
PAPER_PAIRS = [
    ("ViT-B/16", "vitb16_imagenet", "image_noise_late"),
    ("ViT-B/16", "vitb16_imagenet", "uniform_images_late"),
    ("ConvNeXt-B", "convnext_b_imagenet", "label_shuffle_late"),
    ("ConvNeXt-B", "convnext_b_imagenet", "uniform_images_late"),
]

logger = logging.getLogger(__name__)


def build_and_load_model(architecture, ckpt_path, device):
    from temporal_perturbations.models.scratch.scratch_utils import (
        build_timm_feature_model,
        load_timm_state_dict_from_lightning,
    )

    model = build_timm_feature_model(architecture, device)
    sd = load_timm_state_dict_from_lightning(str(ckpt_path), map_location="cpu")
    model.net.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def resolve_ckpt_path(run_name, epoch):
    base = paths.path("runs_root") / run_name / "checkpoints" / "all_epochs"
    candidates = [
        base / f"epoch_{epoch:03d}.ckpt",
        base / f"epoch_epoch={epoch:03d}.ckpt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No epoch {epoch} checkpoint found in {base}. "
        f"Tried: {[c.name for c in candidates]}"
    )


def extract_embeddings(model, img_dir, batch_size, num_workers, device, architecture):
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from temporal_perturbations.data.eval_transforms import ARCH_EVAL_PREPROCESSING
    from tqdm import tqdm

    preprocessing = ARCH_EVAL_PREPROCESSING[architecture]
    transform = preprocessing.build(224)
    logger.info("Using %s preprocessing (tag: %s)", architecture, preprocessing.tag)

    dataset = ImageFolder(img_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    all_embeddings = []
    all_class_indices = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for images, class_indices in tqdm(loader, desc="Extracting embeddings"):
            emb = model(images.to(device))
            all_embeddings.append(emb.cpu())
            # Copy the labels out of the loader's pinned buffer and drop the batch:
            # a live reference to a pinned tensor stops the pinned pool recycling.
            all_class_indices.append(class_indices.numpy().copy())
            del images, class_indices

    embeddings = torch.cat(all_embeddings).numpy()
    class_indices = np.concatenate(all_class_indices)
    synset_ids = [dataset.classes[i] for i in class_indices]
    logger.info("Extracted %d embeddings (%d-d)", *embeddings.shape)
    return embeddings, synset_ids


def extract_and_cache(run_name, epoch, architecture, cache_dir, img_dir,
                      device, batch_size, num_workers, force=False,
                      synsets_dir=SYNSETS_DIR):
    cache_dir = Path(cache_dir)
    emb_path = cache_dir / f"{run_name}_epoch{epoch}.npy"
    synsets_path = Path(synsets_dir) / f"{run_name}_epoch{epoch}_synsets.json"

    if emb_path.exists() and synsets_path.exists() and not force:
        logger.info("Loading cached embeddings from %s", emb_path)
        embeddings = np.load(emb_path)
        with synsets_path.open() as f:
            synset_ids = json.load(f)
        return embeddings, synset_ids

    ckpt_path = resolve_ckpt_path(run_name, epoch)
    logger.info("Loading %s from %s", architecture, ckpt_path)
    model = build_and_load_model(architecture, ckpt_path, device)

    embeddings, synset_ids = extract_embeddings(
        model, str(img_dir), batch_size, num_workers, device, architecture,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    synsets_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    with synsets_path.open("w") as f:
        json.dump(synset_ids, f)
    logger.info("Cached %d embeddings (%d-d) to %s", *embeddings.shape, emb_path)

    del model
    torch.cuda.empty_cache()

    return embeddings, synset_ids


def load_imagenet_category_map(synset_ids):
    """Map ImageNet synset IDs to superordinate categories via WordNet.

    Uses word-level matching (all noun senses of each category word) so
    that e.g. food.n.01 and food.n.02 both count as "Food".  Dictionary
    order controls first-match priority: food before plant_part (fruits
    go to food), vehicle before sports_equipment (golfcart goes to
    vehicle).
    """
    from nltk.corpus import wordnet as wn

    CATEGORY_WORDS = {
        'Animal':             'animal',
        'Vehicle':            'vehicle',
        'Clothing':           'clothing',
        'Food':               'food',
        'Musical Instrument': 'musical_instrument',
        'Plant Part':         'plant_part',
        'Structure':          'structure',
        'Tool':               'tool',
        'Furnishing':         'furnishing',
        'Sports Equipment':   'sports_equipment',
        'Appliance':          'appliance',
        'Geological Formation': 'geological_formation',
        'Utensil':            'utensil',
        'Electronic Equipment': 'electronic_equipment',
        'Measuring Instrument': 'measuring_instrument',
    }

    cat_synsets = {
        label: set(wn.synsets(word, pos='n'))
        for label, word in CATEGORY_WORDS.items()
    }

    synset_to_cat = {}
    for sid in set(synset_ids):
        syn = wn.synset_from_pos_and_offset('n', int(sid[1:]))
        ancestors = set()
        for path in syn.hypernym_paths():
            ancestors.update(path)
        ancestors.add(syn)
        cat = "Uncategorized"
        for label, syns in cat_synsets.items():
            if syns & ancestors:
                cat = label
                break
        synset_to_cat[sid] = cat

    categories = [synset_to_cat[sid] for sid in synset_ids]

    cat_names = list(CATEGORY_WORDS.keys())
    PALETTE_COLORS = {
        'Animal':               '#1f77b4',
        'Vehicle':              '#e377c2',
        'Clothing':             '#ff7f0e',
        'Food':                 '#2ca02c',
        'Musical Instrument':   '#9467bd',
        'Plant Part':           '#17becf',
        'Structure':            '#bcbd22',
        'Tool':                 '#7f7f7f',
        'Furnishing':           '#d62728',
        'Sports Equipment':     '#8c564b',
        'Appliance':            '#aec7e8',
        'Geological Formation': '#ffbb78',
        'Utensil':              '#98df8a',
        'Electronic Equipment': '#ff9896',
        'Measuring Instrument': '#c5b0d5',
        'Uncategorized':        '#bfbfbf',
    }
    palette = {c: PALETTE_COLORS[c] for c in cat_names + ['Uncategorized']}

    counts = pd.Series(categories).value_counts()
    logger.info("ImageNet category distribution:\n%s", counts.to_string())

    return categories, palette


def run_trimap(embeddings, random_state=42, **kwargs):
    import trimap

    reducer = trimap.TRIMAP(
        n_dims=2,
        n_inliers=kwargs.get("n_inliers", 12),
        n_outliers=kwargs.get("n_outliers", 4),
        n_random=kwargs.get("n_random", 3),
        n_iters=kwargs.get("n_iters", 400),
        verbose=True,
    )
    np.random.seed(random_state)
    return reducer.fit_transform(embeddings)


def plot_side_by_side(coords_baseline, coords_perturbed, categories, palette,
                      title_baseline, title_perturbed, output_path,
                      suptitle=None):
    import matplotlib.pyplot as plt

    cat_order = [c for c in palette if c != "Uncategorized"] + ["Uncategorized"]
    cat_indices = {c: [] for c in cat_order}
    for i, c in enumerate(categories):
        cat_indices[c].append(i)

    n_points = len(categories)
    point_size = 2
    alpha = 1.0

    fig, (ax_b, ax_p) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, coords, title in [
        (ax_b, coords_baseline, title_baseline),
        (ax_p, coords_perturbed, title_perturbed),
    ]:
        for cat in cat_order:
            idx = cat_indices[cat]
            if not idx:
                continue
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                c=[palette[cat]],
                s=point_size,
                alpha=alpha,
                label=cat,
                edgecolors="none",
                rasterized=True,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles, labels = ax_b.get_legend_handles_labels()
    legend_handles = []
    for h in handles:
        legend_handles.append(plt.scatter(
            [], [], c=[h.get_facecolors()[0][:3]], s=60, alpha=1.0,
            edgecolors="none",
        ))
    fig.legend(
        legend_handles, labels,
        loc="lower center",
        ncol=min(len(labels), 8),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    if suptitle is None:
        suptitle = f"TriMap ({n_points} ImageNet val images)"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    for ext in ("png", "pdf"):
        out = output_path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out)
    plt.close(fig)


def draw_pair(args, architecture, run_baseline, run_perturbed, epoch, device):
    """Extract (or load) both runs' features, TriMap them, save coords and the plot."""
    perturbation = run_perturbed.replace(run_baseline.rsplit("_baseline", 1)[0] + "_", "")
    arch_short = architecture.replace("/", "").replace("-", "").lower()
    stem = args.stem or (
        f"trimap_{arch_short}_baseline_vs_{perturbation}_epoch{epoch}"
    )

    logger.info("=== Baseline: %s epoch %d ===", run_baseline, epoch)
    emb_baseline, ids_baseline = extract_and_cache(
        run_baseline, epoch, architecture,
        cache_dir=args.cache_dir, img_dir=args.imagenet_dir,
        device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, force=args.no_cache,
    )

    logger.info("=== Perturbed: %s epoch %d ===", run_perturbed, epoch)
    emb_perturbed, ids_perturbed = extract_and_cache(
        run_perturbed, epoch, architecture,
        cache_dir=args.cache_dir, img_dir=args.imagenet_dir,
        device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, force=args.no_cache,
    )

    if args.cache_only:
        logger.info("Cache-only mode; skipping TriMap and plotting.")
        return

    assert ids_baseline == ids_perturbed, (
        "Image order mismatch between baseline and perturbed extractions"
    )

    logger.info("Loading ImageNet WordNet category mapping...")
    categories, palette = load_imagenet_category_map(ids_baseline)

    feat_dim = emb_baseline.shape[1]
    suptitle = (
        f"TriMap of {architecture} {feat_dim}-d Penultimate Features "
        f"({len(ids_baseline)} ImageNet val images)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.coords_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / stem

    coords_baseline_path = args.coords_dir / f"{stem}_coords_baseline.npy"
    coords_perturbed_path = args.coords_dir / f"{stem}_coords_perturbed.npy"

    if args.replot and coords_baseline_path.exists() and coords_perturbed_path.exists():
        logger.info("Loading cached TriMap coordinates from %s", args.coords_dir)
        coords_baseline = np.load(coords_baseline_path)
        coords_perturbed = np.load(coords_perturbed_path)
    else:
        if args.standardize:
            from sklearn.preprocessing import StandardScaler
            logger.info("Standardizing embeddings...")
            scaler = StandardScaler().fit(emb_baseline)
            emb_baseline = scaler.transform(emb_baseline)
            emb_perturbed = scaler.transform(emb_perturbed)

        trimap_kwargs = dict(
            n_inliers=args.n_inliers, n_outliers=args.n_outliers,
            n_random=args.n_random, n_iters=args.n_iters,
        )

        logger.info("Running TriMap on baseline (%d x %d)...", *emb_baseline.shape)
        coords_baseline = run_trimap(emb_baseline, random_state=args.random_state, **trimap_kwargs)

        logger.info("Running TriMap on perturbed (%d x %d)...", *emb_perturbed.shape)
        coords_perturbed = run_trimap(emb_perturbed, random_state=args.random_state, **trimap_kwargs)

        np.save(coords_baseline_path, coords_baseline)
        np.save(coords_perturbed_path, coords_perturbed)
        logger.info("Saved TriMap coordinates to %s", args.coords_dir)

    pert_label = perturbation.replace("_", " ").title()
    plot_side_by_side(
        coords_baseline, coords_perturbed,
        categories, palette,
        title_baseline=f"Baseline (epoch {epoch})",
        title_perturbed=f"{pert_label} (epoch {epoch})",
        output_path=output_path,
        suptitle=suptitle,
    )


def main():
    parser = argparse.ArgumentParser(
        description="TriMap visualization of cross-architecture penultimate features. "
                    "With no run named, draws the four maps Fig 3 reads "
                    "(honors TP_SEED_SUFFIX)."
    )
    parser.add_argument("--architecture", default=None,
                        choices=["ViT-B/16", "ConvNeXt-B", "RN50"],
                        help="Architecture family (omit, with the run flags, for the Fig 3 set).")
    parser.add_argument("--run-baseline", default=None,
                        help="Baseline run directory name under runs_root/.")
    parser.add_argument("--run-perturbed", default=None,
                        help="Perturbed run directory name under runs_root/.")
    parser.add_argument("--epoch", type=int, default=PAPER_EPOCH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--coords-dir", type=Path, default=COORDS_DIR)
    parser.add_argument("--stem", default=None,
                        help="Output stem for a single named pair (default derived from the run names).")
    parser.add_argument("--imagenet-dir", type=Path, default=None,
                        help="ImageNet val ImageFolder (default <imagenet_root>/val).")
    parser.add_argument("--standardize", action="store_true",
                        help="Z-score each dimension before TriMap.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-inliers", type=int, default=12)
    parser.add_argument("--n-outliers", type=int, default=4)
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=400)
    parser.add_argument("--replot", action="store_true",
                        help="Replot from cached TriMap coordinates (skip TriMap).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.imagenet_dir = args.imagenet_dir or paths.path("imagenet_root") / "val"
    device = torch.device(args.device)

    named = (args.architecture, args.run_baseline, args.run_perturbed)
    if any(named):
        if not all(named):
            parser.error("--architecture, --run-baseline and --run-perturbed go together")
        pairs = [named]
    else:
        if args.stem:
            parser.error("--stem applies to a single named pair")
        pairs = [(arch, f"{prefix}_baseline{SEED_SUFFIX}", f"{prefix}_{pert}{SEED_SUFFIX}")
                 for arch, prefix, pert in PAPER_PAIRS]

    for architecture, run_baseline, run_perturbed in pairs:
        draw_pair(args, architecture, run_baseline, run_perturbed, args.epoch, device)
    logger.info("Done.")


if __name__ == "__main__":
    main()
