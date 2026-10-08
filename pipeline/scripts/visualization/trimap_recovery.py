"""Compute per-snapshot TriMap coordinates for the Fig 4 recovery filmstrips.

Extracts penultimate features on ImageNet val for 10 (run, epoch) pairs across
two case studies (Case T: ViT-B/16 late image noise; Case L: CLIP-ViT-B/32
early uniform images), then runs TriMap independently on each to produce 2-D
coordinates for the scatter-plot filmstrip in Panel B.

Writes `<results_root>/trimap/trimap_fig5b_{run}_epoch{E}_coords.npy`, read by
fig4_recovery_trajectories.py. Features are cached in
`<feature_cache>/cross_arch_embeddings/`, image synset order in
`<results_root>/imagenet_embeddings/`.
"""
import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

VIZ_DIR = Path(__file__).resolve().parent
PIPELINE = VIZ_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths
from temporal_perturbations.data.eval_transforms import ARCH_EVAL_PREPROCESSING
from temporal_perturbations.models.scratch.scratch_utils import (
    build_cross_arch_model,
    sorted_scratch_checkpoints,
)

CACHE_DIR = paths.path("feature_cache") / "cross_arch_embeddings"
SYNSETS_DIR = paths.results_root() / "imagenet_embeddings"
COORDS_DIR = paths.results_root() / "trimap"

logger = logging.getLogger(__name__)

JOBS = [
    # Case T: ViT-B/16, image_noise_late, onset=70, total=90
    {"run": "vitb16_imagenet_baseline",         "epoch": 70, "arch": "ViT-B/16"},
    {"run": "vitb16_imagenet_image_noise_late",  "epoch": 70, "arch": "ViT-B/16"},
    {"run": "vitb16_imagenet_image_noise_late",  "epoch": 75, "arch": "ViT-B/16"},
    {"run": "vitb16_imagenet_image_noise_late",  "epoch": 89, "arch": "ViT-B/16"},
    {"run": "vitb16_imagenet_baseline",         "epoch": 89, "arch": "ViT-B/16"},
    # Case L: CLIP-ViT-B/32, uniform_images_early, onset=5, total=32
    {"run": "clip_vitb32_baseline",              "epoch":  5, "arch": "CLIP ViT-B/32"},
    {"run": "clip_vitb32_uniform_images_early",  "epoch":  5, "arch": "CLIP ViT-B/32"},
    {"run": "clip_vitb32_uniform_images_early",  "epoch": 10, "arch": "CLIP ViT-B/32"},
    {"run": "clip_vitb32_uniform_images_early",  "epoch": 31, "arch": "CLIP ViT-B/32"},
    {"run": "clip_vitb32_baseline",              "epoch": 31, "arch": "CLIP ViT-B/32"},
]


def resolve_ckpt_path(run_name, epoch):
    base = paths.path("runs_root") / run_name / "checkpoints" / "all_epochs"
    for name in [f"epoch_{epoch:03d}.ckpt", f"epoch_epoch={epoch:03d}.ckpt"]:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No epoch {epoch} checkpoint in {base}")


def extract_embeddings(model, architecture, img_dir, batch_size, num_workers, device):
    preprocessing = ARCH_EVAL_PREPROCESSING[architecture]
    transform = preprocessing.build(224)
    dataset = ImageFolder(str(img_dir), transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    all_emb, all_idx = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        for images, class_indices in loader:
            images = images.to(device)
            emb = model(images)
            all_emb.append(emb.float().cpu())
            all_idx.append(class_indices)

    embeddings = torch.cat(all_emb).numpy()
    class_indices = torch.cat(all_idx).numpy()
    synset_ids = [dataset.classes[i] for i in class_indices]
    logger.info("Extracted %d embeddings (%d-d)", *embeddings.shape)
    return embeddings, synset_ids


def extract_and_cache(run_name, epoch, architecture, model_state, device,
                      batch_size, num_workers, img_dir, force=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SYNSETS_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = CACHE_DIR / f"{run_name}_epoch{epoch}.npy"
    syn_path = SYNSETS_DIR / f"{run_name}_epoch{epoch}_synsets.json"

    if emb_path.exists() and syn_path.exists() and not force:
        logger.info("Cached: %s", emb_path.name)
        embeddings = np.load(emb_path)
        with syn_path.open() as f:
            synset_ids = json.load(f)
        return embeddings, synset_ids

    ckpt_path = resolve_ckpt_path(run_name, epoch)
    logger.info("Loading %s epoch %d from %s", architecture, epoch, ckpt_path.name)
    model = model_state["load_weights"](str(ckpt_path))
    model_state["current_model"] = model

    embeddings, synset_ids = extract_embeddings(
        model, architecture, img_dir, batch_size, num_workers, device,
    )

    np.save(emb_path, embeddings)
    with syn_path.open("w") as f:
        json.dump(synset_ids, f)
    logger.info("Saved %s (%d x %d)", emb_path.name, *embeddings.shape)
    return embeddings, synset_ids


def run_trimap(embeddings, random_state=42):
    import trimap
    reducer = trimap.TRIMAP(
        n_dims=2, n_inliers=12, n_outliers=4, n_random=3,
        n_iters=400, verbose=True,
    )
    np.random.seed(random_state)
    return reducer.fit_transform(embeddings)


def trimap_coord_path(run_name, epoch):
    return COORDS_DIR / f"trimap_fig5b_{run_name}_epoch{epoch}_coords.npy"


def main():
    parser = argparse.ArgumentParser(description="Compute the Fig 4 recovery TriMap coordinates.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--imagenet-dir", type=Path, default=None,
                        help="ImageNet val ImageFolder (default <imagenet_root>/val).")
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached.")
    parser.add_argument("--cache-only", action="store_true", help="Extract embeddings only, skip TriMap.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seed-suffix", default=os.environ.get("TP_SEED_SUFFIX", ""),
                        help='replicate family, appended to every run name (e.g. "_seed43"; '
                             'default $TP_SEED_SUFFIX)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    device = torch.device(args.device)
    args.imagenet_dir = args.imagenet_dir or paths.path("imagenet_root") / "val"
    COORDS_DIR.mkdir(parents=True, exist_ok=True)

    current_arch = None
    model_state = {"load_weights": None, "current_model": None}

    for job in JOBS:
        run, epoch, arch = job["run"] + args.seed_suffix, job["epoch"], job["arch"]
        logger.info("=== %s epoch %d (%s) ===", run, epoch, arch)

        if arch != current_arch:
            if model_state["current_model"] is not None:
                del model_state["current_model"]
                model_state["current_model"] = None
                gc.collect()
                torch.cuda.empty_cache()

            logger.info("Building %s model...", arch)
            model_obj, load_fn = build_cross_arch_model(arch, device, clip_feature="cls")
            model_state["load_weights"] = load_fn
            model_state["current_model"] = model_obj
            current_arch = arch

        embeddings, _ = extract_and_cache(
            run, epoch, arch, model_state, device,
            args.batch_size, args.num_workers, args.imagenet_dir,
            force=args.force,
        )

        if args.cache_only:
            continue

        coord_path = trimap_coord_path(run, epoch)
        if coord_path.exists() and not args.force:
            logger.info("TriMap coords exist: %s", coord_path.name)
            continue

        logger.info("Running TriMap on %d x %d...", *embeddings.shape)
        coords = run_trimap(embeddings, random_state=args.random_state)
        np.save(coord_path, coords)
        logger.info("Saved %s", coord_path.name)

    if model_state["current_model"] is not None:
        del model_state["current_model"]
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
