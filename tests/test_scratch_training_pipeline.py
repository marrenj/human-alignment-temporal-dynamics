"""
Integration test for the end-to-end scratch-training pipeline.

Uses real images from the image directories:
  - ImageNet training/val: 2 randomly chosen synsets, 5 images each split
    (symlinked into a temp dir so the full dataset is never copied)
  - THINGS behavioral RSA: all 48 images from the real inference CSV / image dir

The test is skipped automatically when the image directories are not present
(e.g. on a machine that doesn't have the datasets mounted).

Run with:
    pytest tests/test_scratch_training_pipeline.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_perturbations.training.trainer import run_training_experiment


from temporal_perturbations import paths


def _configured(key):
    try:
        return paths.path(key)
    except KeyError:
        return Path("/nonexistent")


IMAGENET_ROOT  = _configured("imagenet_root")
THINGS_IMG_DIR = _configured("things_images")
THINGS_CSV     = ROOT / "pipeline" / "data" / "spose_embedding66d_rescaled_48val_reordered.csv"

_PATHS_AVAILABLE = (
    IMAGENET_ROOT.is_dir()
    and THINGS_IMG_DIR.is_dir()
    and THINGS_CSV.is_file()
)

skip_if_no_data = pytest.mark.skipif(
    not _PATHS_AVAILABLE,
    reason="Image directories not found on this machine.",
)


# Helper: build a minimal ImageNet-style temp dir via symlinks

def _symlink_imagenet_subset(
    tmp_root: Path,
    imagenet_root: Path,
    n_classes: int = 2,
    images_per_class: int = 5,
    seed: int = 42,
) -> Path:
    """
    Create a minimal ImageNet directory under tmp_root by symlinking
    n_classes randomly chosen synsets with up to images_per_class images
    per split (train / val).  Returns tmp_root.
    """
    rng = random.Random(seed)

    train_src = imagenet_root / "train"
    val_src   = imagenet_root / "val"

    all_synsets = sorted(p.name for p in train_src.iterdir() if p.is_dir())
    chosen = rng.sample(all_synsets, n_classes)

    for split_src, split_name in [(train_src, "train"), (val_src, "val")]:
        for synset in chosen:
            dst_dir = tmp_root / split_name / synset
            dst_dir.mkdir(parents=True, exist_ok=True)
            images = sorted((split_src / synset).iterdir())[:images_per_class]
            for img in images:
                (dst_dir / img.name).symlink_to(img.resolve())

    return tmp_root


# Test

@skip_if_no_data
@patch("temporal_perturbations.training.trainer.wandb")
def test_scratch_pipeline_runs_one_epoch(mock_wandb: MagicMock, tmp_path: Path, capsys) -> None:
    """
    Train ResNet50 from scratch for 1 epoch on a 2-class ImageNet subset,
    running behavioral RSA at each epoch using all 48 real THINGS images.

    wandb is mocked so the test runs fully offline.
    """
    mock_wandb.init.return_value = MagicMock()

    # Build a tiny ImageNet subset (2 synsets × 5 images) in a temp dir
    img_dir = _symlink_imagenet_subset(
        tmp_root=tmp_path / "imagenet",
        imagenet_root=IMAGENET_ROOT,
        n_classes=2,
        images_per_class=5,
        seed=42,
    )

    save_path = tmp_path / "run"

    config = {
        # mode
        "training_mode": "scratch",
        # paths
        "img_dir": str(img_dir),
        "save_path": str(save_path),
        # model
        "architecture": "RN50",
        "pretrained": False,
        "num_classes": 2,
        # training
        "dataset_type": "imagenet",
        "max_duration": 5,
        "batch_size": 4,
        "criterion": "CrossEntropyLoss",
        "early_stopping_patience": 5,
        "random_seed": 42,
        "cuda": -1,
        # optimizer / scheduler
        "opt": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lr_scheduler": None,
        "lr_warmup_duration": 0,
        # perturbations (none)
        "perturb_type": "none",
        "perturb_epoch": 0,
        "perturb_length": 0,
        "perturb_seed": 42,
        # W&B – run fully offline
        "wandb_project": None,
        "wandb_entity": None,
        # Disable multiprocessing workers in tests (not needed for tiny dataset)
        "num_workers": 0,
        # Behavioral RSA – real 48-image THINGS inference set
        "behavioral_rsa": True,
        "rsa_annotations_file": str(THINGS_CSV),
        "rsa_things_img_dir": str(THINGS_IMG_DIR),
        "model_rdm_distance_metric": "pearson",
        "rsa_similarity_metric": "spearman",
        "debug_logging": False,
    }

    # capsys.disabled() lets all pipeline output reach the console live
    with capsys.disabled():
        run_training_experiment(config)

    # Verify training results CSV was produced with expected columns
    results_csv = next(save_path.rglob("training_res.csv"))
    lines = results_csv.read_text().strip().splitlines()
    assert len(lines) >= 2, "Expected header + at least one epoch row"

    header = lines[0].split(",")
    assert "epoch"      in header
    assert "train_loss" in header
    assert "val_loss"   in header
