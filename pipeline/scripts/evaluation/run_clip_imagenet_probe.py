"""ImageNet-1k linear probe for CLIP-ViT-B/32 checkpoints, along a training trajectory.

Fits a 1000-class logistic probe (L-BFGS on GPU) on z-scored ImageNet-1k train
features, selects C with the repo-wide cold-started sweep, then scores ImageNet-1k val.
Writes one JSON per checkpoint to
`<results_root>/imagenet_ood_eval_scratch_clip_cls/<run>/ood_probe_eval_epoch{E}.json`
(`imagenet_ood_eval_scratch/` for `--clip_feature projected`), the record the figures read.

The probe fit (`_fit_probe_torch`, `_select_C`, `_score_topk`) is shared with
`run_clip_imagenet_ood_probe.py`, which writes the same record plus the OOD variants
at a few epochs per run; this script covers the denser epoch grid the recovery
curves need. Without --force, a current-recipe record from either driver is reused.

Usage
-----
    # Every CLIP run, at the epochs given:
    python pipeline/scripts/evaluation/run_clip_imagenet_probe.py --epochs 7,8,10,13,16,20,24,28,29,30

    # One run:
    python pipeline/scripts/evaluation/run_clip_imagenet_probe.py \\
        --run_name clip_vitb32_baseline --epochs 1,2,3,8,10,16,20,24,30

    # An explicit checkpoint:
    python pipeline/scripts/evaluation/run_clip_imagenet_probe.py \\
        --checkpoint /path/to/epoch_031.ckpt --run_name my_run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP_DIR))

from temporal_perturbations import paths  # noqa: E402
from temporal_perturbations.inference.linear_probe_eval import (  # noqa: E402
    PROBE_C_SELECTION,
    PROBE_FEATURE_SCALING,
    fit_standardizer,
    is_current_recipe,
)
from temporal_perturbations.models.clip_yfcc import (  # noqa: E402
    CLIP_FEATURES,
    DEFAULT_CLIP_FEATURE,
    load_clip_feature_model,
)
from temporal_perturbations.models.scratch.scratch_utils import (  # noqa: E402
    parse_epoch_from_scratch_filename,
    sorted_scratch_checkpoints,
)

from run_clip_imagenet_ood_probe import (  # noqa: E402
    CLIP_BASELINE,
    DEFAULT_OUTPUT_DIR,
    PROBE_FINAL_MAX_ITER,
    _extract_features_imagenet,
    _fit_probe_torch,
    _score_topk,
    _select_C,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CLIP_RUN_PREFIX = "clip_vitb32"


def build_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def _jobs(args, runs_root):
    """[(run_name, epoch, checkpoint), ...] for every requested run and epoch."""
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        return [(args.run_name or ckpt.parent.parent.parent.name,
                 parse_epoch_from_scratch_filename(ckpt.name), ckpt)]
    if args.run_name:
        names = [args.run_name]
    else:
        names = sorted(d.name for d in runs_root.iterdir()
                       if d.is_dir() and (d.name == CLIP_BASELINE
                                          or d.name.startswith(CLIP_RUN_PREFIX)))
    jobs = []
    for name in names:
        avail = dict(sorted_scratch_checkpoints(runs_root / name / "checkpoints" / "all_epochs"))
        epochs = ([int(e) for e in args.epochs.split(",")] if args.epochs else [max(avail)])
        jobs += [(name, e, avail[e]) for e in epochs if e in avail]
    return jobs


def probe_one(ckpt_path, run_name, epoch, args, transform, device, logger):
    output_root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    if args.clip_feature == "cls":
        output_root = output_root.parent / (output_root.name + "_clip_cls")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"ood_probe_eval_epoch{epoch}.json"

    if out_file.exists() and not args.force:
        with open(out_file) as f:
            cached = json.load(f)
        if is_current_recipe(cached) and "imagenet_val" in cached:
            logger.info("[skip] %s (cached)", out_file)
            return
    logger.info("checkpoint: %s -> %s", ckpt_path.name, out_file)
    if args.dry_run:
        return

    # The same loader as run_clip_imagenet_ood_probe.py, since both write the same record.
    model = load_clip_feature_model(str(ckpt_path), clip_feature=args.clip_feature,
                                    device=str(device))

    feat_cache_dir = Path(args.feature_cache)
    train_feat_path = feat_cache_dir / f"{run_name}_{ckpt_path.stem}_{args.clip_feature}_train.npz"
    if train_feat_path.exists() and not args.force:
        logger.info("loading cached train features from %s", train_feat_path)
        data = np.load(train_feat_path)
        train_feats, train_labels = data["feats"], data["labels"]
    else:
        logger.info("extracting ImageNet-1k train features...")
        train_feats, train_labels = _extract_features_imagenet(
            model, args.imagenet_train, transform, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            logger=logger, label="ImageNet train",
        )
        train_feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(train_feat_path, feats=train_feats.astype(np.float32),
                            labels=train_labels)

    n_features = train_feats.shape[1]
    const_mask = train_feats.std(axis=0) == 0
    n_const = int(const_mask.sum())
    keep = None
    if n_const > 0:
        keep = ~const_mask
        train_feats = train_feats[:, keep]
        logger.info("  dropped %d/%d constant dims; %d remain",
                    n_const, n_features, n_features - n_const)
    standardize = fit_standardizer(train_feats)
    train_feats = standardize(train_feats)

    best_c, sweep_info = _select_C(train_feats, train_labels, device, logger)
    logger.info("fitting final probe on full ImageNet train...")
    W, b_vec, classes, fit_info = _fit_probe_torch(
        train_feats, train_labels, best_c, PROBE_FINAL_MAX_ITER, device, logger,
    )
    if not fit_info["converged"]:
        logger.warning("[not-converged] probe stopped at %d iters", fit_info["n_iter"])
    del train_feats, train_labels
    gc.collect()

    logger.info("extracting ImageNet-1k val features...")
    val_feats, val_labels = _extract_features_imagenet(
        model, args.imagenet_val, transform, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        logger=logger, label="ImageNet val",
    )
    if keep is not None:
        val_feats = val_feats[:, keep]
    val_feats = standardize(val_feats)
    imagenet_top1, imagenet_top5 = _score_topk(W, b_vec, classes, val_feats, val_labels)
    n_val = len(val_labels)
    logger.info("ImageNet-1k val: top-1 = %.4f  top-5 = %.4f  (%d images)",
                imagenet_top1, imagenet_top5, n_val)
    del val_feats, val_labels, model
    gc.collect()
    torch.cuda.empty_cache()

    record = {
        "architecture": "CLIP ViT-B/32",
        "run_name": run_name,
        "epoch": epoch,
        "checkpoint": ckpt_path.name,
        "clip_feature": args.clip_feature,
        "evaluation_type": "imagenet_linear_probe",
        "train_dataset": "imagenet1k_train",
        "n_features": n_features,
        "n_constant_dims_dropped": n_const,
        "c_selection": PROBE_C_SELECTION,
        "feature_scaling": PROBE_FEATURE_SCALING,
        "c_sweep": sweep_info,
        **fit_info,
        "imagenet_val": {
            "top1_accuracy": imagenet_top1,
            "top5_accuracy": imagenet_top5,
            "n_test": n_val,
        },
    }
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2)
    logger.info("saved %s", out_file)
    train_feat_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", type=Path, default=None, help="Default: paths key runs_root.")
    p.add_argument("--run_name", default=None,
                   help="One run under runs_root. Default: the CLIP baseline and "
                        "every clip_vitb32_* run.")
    p.add_argument("--epochs", default=None,
                   help="Comma-separated epochs. Default: each run's final epoch.")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit Lightning .ckpt; bypasses runs_root discovery.")
    p.add_argument("--imagenet_train", default=None, help="Default: <imagenet_root>/train.")
    p.add_argument("--imagenet_val", default=None, help="Default: <imagenet_root>/val.")
    p.add_argument("--output_dir", default=None,
                   help=f"Default: {DEFAULT_OUTPUT_DIR.name}_clip_cls under results_root "
                        f"({DEFAULT_OUTPUT_DIR.name} for --clip_feature projected).")
    p.add_argument("--feature_cache", default=str(paths.path("feature_cache") / "clip_imagenet_probe"),
                   help="Train features are cached here until the probe is written.")
    p.add_argument("--clip_feature", default=DEFAULT_CLIP_FEATURE, choices=sorted(CLIP_FEATURES),
                   help="'cls' (768-d, pre-projection; default) or 'projected' "
                        "(512-d, post-projection, L2-normalized).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = p.parse_args()
    args.imagenet_train = args.imagenet_train or paths.path("imagenet_root") / "train"
    args.imagenet_val = args.imagenet_val or paths.path("imagenet_root") / "val"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("imagenet_probe")

    runs_root = args.runs_root or paths.path("runs_root")
    jobs = _jobs(args, runs_root)
    logger.info("%d checkpoint(s) to probe", len(jobs))
    device = torch.device(args.device)
    transform = build_transform()
    for run_name, epoch, ckpt in jobs:
        probe_one(Path(ckpt), run_name, epoch, args, transform, device, logger)


if __name__ == "__main__":
    main()
