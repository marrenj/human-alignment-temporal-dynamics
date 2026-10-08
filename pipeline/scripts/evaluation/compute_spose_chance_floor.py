"""Chance-level SPoSE alignment from randomly initialized (untrained) models.

For each of the four cross-architecture families (RN50, ViT-B/16, ConvNeXt-B,
CLIP ViT-B/32), build the model with random weights (no checkpoint loaded),
extract 1854-image THINGS embeddings, build the Pearson-distance RDM, and
compute Spearman correlation against the human SPoSE RDM.  Repeated over
multiple random seeds to give a mean +/- std floor per architecture.

The output JSON (results_root/spose_chance_floor.json) is read by
visualization/fig3_storyboard.py to draw a per-panel dashed chance line.

Usage:
    python pipeline/scripts/evaluation/compute_spose_chance_floor.py
    python pipeline/scripts/evaluation/compute_spose_chance_floor.py --seeds 10
    python pipeline/scripts/evaluation/compute_spose_chance_floor.py --device cpu
"""

from pathlib import Path
import argparse
import json
import sys

EXP_DIR  = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from temporal_perturbations import paths
from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset
from temporal_perturbations.data.eval_transforms import (
    ARCH_EVAL_PREPROCESSING,
    CLIP_CONTRASTIVE,
)
from temporal_perturbations.models.scratch.scratch_utils import (
    TimmFeatureWrapper,
    build_timm_feature_model,
)
from temporal_perturbations.models.factory import _TIMM_ARCH_MAP
from scipy.stats import spearmanr
from temporal_perturbations.inference.inference_core import (
    compute_model_rdm,
    build_spose_target_rdm,
)

ARCHES = ["RN50", "ViT-B/16", "ConvNeXt-B", "CLIP ViT-B/32"]

DEFAULT_CONFIG = PIPELINE / "configs" / "inference_config_posthoc_1854.yaml"
DEFAULT_ANNOTATIONS = PIPELINE / "data" / "spose_embedding66d_rescaled.csv"
DEFAULT_OUTPUT = paths.results_root() / "spose_chance_floor.json"


def _parse_args():
    p = argparse.ArgumentParser(description="Compute chance-level SPoSE alignment from random-init models.")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of random initializations per architecture (default: 5).")
    p.add_argument("--img_dir", type=str, default=None, help="Default: paths key things_images.")
    p.add_argument("--annotations", type=str, default=str(DEFAULT_ANNOTATIONS))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    args.img_dir = str(args.img_dir or paths.path("things_images"))
    return args


def _preload_images(img_dir, annotations, preprocessing, batch_size, num_workers):
    ds = ThingsBehavioralDataset(
        img_annotations_file=str(annotations),
        img_dir=img_dir,
        transform=preprocessing.build(224),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    names, chunks = [], []
    for image_names, images, _ in loader:
        names.extend(list(image_names))
        chunks.append(images)
    return torch.cat(chunks, dim=0), names


def _embed(model, images_cpu, batch_size, device):
    use_autocast = device.type == "cuda"
    out = []
    with torch.no_grad():
        for i in range(0, images_cpu.shape[0], batch_size):
            chunk = images_cpu[i:i + batch_size].to(device, non_blocking=False)
            if use_autocast:
                with torch.amp.autocast("cuda"):
                    feats = model(chunk)
            else:
                feats = model(chunk)
            out.append(feats.float().cpu())
    return torch.cat(out, dim=0).numpy()


def _random_timm_model(architecture, seed, device):
    """Build a randomly-initialized timm model (no pretrained weights)."""
    import timm
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = timm.create_model(_TIMM_ARCH_MAP[architecture], pretrained=False, num_classes=1000)
    model = TimmFeatureWrapper(net).to(device)
    model.eval()
    return model


def _random_clip_model(seed, device, clip_feature="cls"):
    """Build a randomly-initialized CLIP ViT-B/32 (no checkpoint loaded)."""
    from temporal_perturbations.models.clip_yfcc.model import CLIP
    from temporal_perturbations.models.clip_yfcc.model_loader import load_model_config
    from temporal_perturbations.models.clip_yfcc.model import (
        CLIPVisionCLSExtractor,
        CLIPVisionProjectedExtractor,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = load_model_config("vit_b_32")
    clip_model = CLIP(cfg)
    clip_model.to(device).eval()

    extractor = CLIPVisionCLSExtractor if clip_feature == "cls" else CLIPVisionProjectedExtractor
    return extractor(clip_model).to(device).eval()


def main():
    args = _parse_args()
    device = torch.device(args.device)
    annotations = str(args.annotations)

    image_cache = {}

    results = {}
    for arch in ARCHES:
        is_clip = arch == "CLIP ViT-B/32"
        preprocessing = CLIP_CONTRASTIVE if is_clip else ARCH_EVAL_PREPROCESSING[arch]

        cache_key = preprocessing.tag
        if cache_key not in image_cache:
            print(f"Preloading 1854 THINGS images ({preprocessing.tag})...")
            images, names = _preload_images(
                args.img_dir, annotations, preprocessing,
                args.batch_size, args.num_workers,
            )
            image_cache[cache_key] = (images, names)
        images, names = image_cache[cache_key]

        spose_rdm_upper = build_spose_target_rdm(annotations, "pearson", image_order=names)

        rhos = []
        for seed in range(args.seeds):
            print(f"  {arch} seed {seed}...", end=" ", flush=True)
            if is_clip:
                model = _random_clip_model(seed, device)
            else:
                model = _random_timm_model(arch, seed, device)

            embeddings = _embed(model, images, args.batch_size, device)
            model_rdm = compute_model_rdm(
                embeddings, dataset_name="targets",
                annotations_file=None, distance_metric="pearson",
            )
            model_rdm_upper = model_rdm[np.triu_indices_from(model_rdm, k=1)]
            rho, _ = spearmanr(model_rdm_upper, spose_rdm_upper)
            rhos.append(rho)
            print(f"rho={rho:.4f}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[arch] = {
            "rhos": rhos,
            "mean": float(np.mean(rhos)),
            "std": float(np.std(rhos)),
            "n_seeds": args.seeds,
        }
        print(f"  {arch}: mean={results[arch]['mean']:.4f} +/- {results[arch]['std']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
