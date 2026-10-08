"""Generate CLIP-HBA THINGS 48x66 embeddings at offset epochs for the SPoSE-dimension analysis.

For each perturbation run, loads the DoRA checkpoint at epoch (onset + offset)
and forward-passes the 48 THINGS behavioral images to produce
things_embeddings_epoch{E}.npy, and the matching baseline embeddings at the same
epochs. The model is initialized once; only the DoRA adapter weights are swapped
per checkpoint.

Outputs mirror the checkpoint layout: a run at `<cliphba_runs_root>/<rel>/` writes
`<results_root>/cliphba_embeddings/<rel>/eval_results/things_embeddings_epoch{E}.npy`.

Typical use:
    python pipeline/scripts/evaluation/extract_cliphba_things_embeddings.py --offset 10
    python pipeline/scripts/evaluation/extract_cliphba_things_embeddings.py --offset 10 --perturb random_target
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths  # noqa: E402
from temporal_perturbations.data.spose_dimensions import classnames66
from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset
from temporal_perturbations.models.clip_hba.clip_hba_utils import (
    initialize_cliphba_model,
    load_dora_checkpoint,
)

EMBEDDINGS_ROOT = paths.results_root() / "cliphba_embeddings"


def _ckpt_root() -> Path:
    return paths.path("cliphba_runs_root")
ANNOTATIONS_FILE = REPO_ROOT / "pipeline" / "data" / "spose_embedding66d_rescaled_48val.csv"

PERTURBATIONS = ["random_target", "label_shuffle", "image_noise", "uniform_images"]
SEEDS = [1, 2, 3]

MODEL_CFG = dict(
    backbone_name="ViT-L/14",
    vision_layers=2,
    transformer_layers=1,
    rank=32,
    dora_dropout=0.1,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offset", type=int, default=10,
                   help="Number of epochs after onset to evaluate (default 10; 0 = the onset itself)")
    p.add_argument("--perturb", type=str, nargs="+", default=None,
                   help="Perturbation types to process (default: all four)")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="Seeds to process (default: 1 2 3)")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Torch device (default: cuda:0)")
    p.add_argument("--batch_size", type=int, default=48,
                   help="Batch size for forward pass (48 = all images at once)")
    p.add_argument("--dry_run", action="store_true",
                   help="List what would be generated without running inference")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if embedding file already exists")
    return p.parse_args()


def _discover_onset_epochs() -> list[int]:
    """Return the 28 onset epochs from the filesystem."""
    onsets = set()
    for seed_dir in _ckpt_root().glob("random_target_perturb_seed1/epoch*_length1"):
        m = re.match(r"epoch(\d+)_length1$", seed_dir.name)
        if m:
            onsets.add(int(m.group(1)))
    return sorted(onsets)


def _find_run_dir(perturb_type: str, seed: int, onset: int) -> Path | None:
    parent = _ckpt_root() / f"{perturb_type}_perturb_seed{seed}" / f"epoch{onset}_length1"
    if not parent.exists():
        return None
    runs = list(parent.glob(f"vit_l_14_rank32_perturb-type-{perturb_type}_*"))
    return runs[0] if runs else None


def _baseline_run_dir(seed: int) -> Path:
    return (_ckpt_root() / f"baseline_seed{seed}" /
            f"vit_l_14_rank32_perturb-type-none_init-seed{seed}behavioral-rsa-True")


def _embedding_path(run_dir: Path, epoch: int) -> Path:
    return (EMBEDDINGS_ROOT / run_dir.relative_to(_ckpt_root()) / "eval_results"
            / f"things_embeddings_epoch{epoch}.npy")


def _dora_exists(run_dir: Path, epoch: int) -> bool:
    return (run_dir / "dora_params" / f"epoch{epoch}_dora_params.pth").exists()


import logging  # noqa: E402
logging.basicConfig(level=logging.WARNING)
_logger = logging.getLogger("extract_cliphba_things_embeddings")
_logger.setLevel(logging.INFO)


def build_job_list(args, onsets):
    """Return list of (run_dir, target_epoch, save_path) tuples."""
    offset = args.offset
    perturb_types = args.perturb or PERTURBATIONS
    seeds = args.seeds
    max_epoch = 117

    jobs = []

    needed_baseline_epochs = set()

    for pt in perturb_types:
        for seed in seeds:
            for onset in onsets:
                target_ep = onset + offset
                if target_ep > max_epoch:
                    continue
                run_dir = _find_run_dir(pt, seed, onset)
                if run_dir is None:
                    continue
                save_path = _embedding_path(run_dir, target_ep)
                if save_path.exists() and not args.force:
                    continue
                if not _dora_exists(run_dir, target_ep):
                    continue
                jobs.append((run_dir, target_ep, save_path))
                needed_baseline_epochs.add((seed, target_ep))

    for seed, ep in sorted(needed_baseline_epochs):
        base_dir = _baseline_run_dir(seed)
        save_path = _embedding_path(base_dir, ep)
        if save_path.exists() and not args.force:
            continue
        if not _dora_exists(base_dir, ep):
            continue
        jobs.append((base_dir, ep, save_path))

    return jobs


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Forward-pass all images and return (N, 66) numpy array."""
    all_embs = []
    for batch in dataloader:
        _, images, _ = batch
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            embs = model(images)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0).numpy()


def main():
    args = parse_args()
    onsets = _discover_onset_epochs()
    _logger.info("Found %d onset epochs: %s", len(onsets), onsets)

    jobs = build_job_list(args, onsets)
    _logger.info("Jobs to run: %d", len(jobs))

    if args.dry_run:
        for run_dir, ep, save_path in jobs:
            status = "EXISTS" if save_path.exists() else "GENERATE"
            print(f"  [{status}] {save_path.relative_to(EMBEDDINGS_ROOT)}  (epoch {ep})")
        return

    if not jobs:
        print("Nothing to generate — all embeddings already exist.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _logger.info("Device: %s", device)

    _logger.info("Initializing CLIP-HBA model...")
    model = initialize_cliphba_model(classnames=classnames66, logger=_logger, **MODEL_CFG)
    model = model.to(device)
    model.eval()

    _logger.info("Loading THINGS behavioral dataset (48 images)...")
    dataset = ThingsBehavioralDataset(
        img_annotations_file=str(ANNOTATIONS_FILE),
        img_dir=str(paths.path("things_images")),
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    generated = 0
    for run_dir, target_ep, save_path in tqdm(jobs, desc="Generating embeddings"):
        load_dora_checkpoint(model, checkpoint_root=run_dir, epoch=target_ep,
                             map_location=device)
        matrix = extract_embeddings(model, dataloader, device)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, matrix)
        generated += 1

    _logger.info("Generated %d embedding files.", generated)


if __name__ == "__main__":
    main()
