"""Linear-probe OOD evaluation for CLIP-ViT-B/32 checkpoints.

Fits a 1000-class linear probe on ImageNet-1k train features (1.28M images),
then evaluates on:
  - ImageNet-1k val (50k images) -- top-1 and top-5
  - Each OOD variant (imagenet-a/o/r/sketch/v2) with class-subset masking

The probe is the classification head that CLIP lacks -- once fitted, scoring
works the same way as the timm models' direct-head OOD eval: forward pass
through the full model + head, mask logits to the OOD class subset, argmax.
Features are z-scored with train statistics, and C is selected exactly as for every
other probe in the repo (`linear_probe_eval._select_C`: each candidate cold-started on
an 80/20 split of the train features, capped at 2,000 iterations); the final probe is
refit from zero on the full train split before scoring.

Results: `<results_root>/imagenet_ood_eval_scratch_clip_cls/<run>/ood_probe_eval_epoch{E}.json`
(`imagenet_ood_eval_scratch/` for `--clip_feature projected`).

Usage
-----
    python pipeline/scripts/evaluation/run_clip_imagenet_ood_probe.py
    python pipeline/scripts/evaluation/run_clip_imagenet_ood_probe.py --runs baseline
    python pipeline/scripts/evaluation/run_clip_imagenet_ood_probe.py --epochs 5,6,31
    python pipeline/scripts/evaluation/run_clip_imagenet_ood_probe.py --include-seed43 --run_filter seed43
    python pipeline/scripts/evaluation/run_clip_imagenet_ood_probe.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP_DIR))

from temporal_perturbations import paths  # noqa: E402
from temporal_perturbations.data.eval_transforms import eval_preprocessing  # noqa: E402
from temporal_perturbations.inference.linear_probe_eval import (  # noqa: E402
    PROBE_C_SELECTION,
    PROBE_FEATURE_SCALING,
    PROBE_TOL,
    _select_C as _select_C_shared,
    fit_standardizer,
    is_current_recipe,
)
from temporal_perturbations.models.clip_yfcc.feature_space import (  # noqa: E402
    DEFAULT_CLIP_FEATURE,
    load_clip_feature_model,
)
from temporal_perturbations.models.scratch.scratch_utils import (  # noqa: E402
    sorted_scratch_checkpoints,
)
from temporal_perturbations.utils.run_hparams import onset_epoch as _onset_epoch  # noqa: E402

from run_imagenet_ood_eval import (  # noqa: E402
    OOD_DATASETS,
    _build_ood_dataset,
)

DEFAULT_OUTPUT_DIR = paths.results_root() / "imagenet_ood_eval_scratch"
DEFAULT_FEATURE_CACHE = paths.path("feature_cache") / "probe_features"

CLIP_PREFIX = "clip_vitb32"
CLIP_BASELINE = "clip_vitb32_baseline"
PERTURB_TYPES = ["image_noise", "label_shuffle", "uniform_images"]
PHASES = ["early", "mid", "late"]

IMAGENET_NUM_CLASSES = 1000
IMAGENET_TRAIN_SIZE = 1281167
IMAGENET_VAL_SIZE = 50000

PROBE_FINAL_MAX_ITER = 20000


def _extract_features_imagenet(model, split_dir, transform, device, batch_size,
                               num_workers, logger, label=""):
    """Extract features from a local ImageNet split (train or val)."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(split_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True,
                        prefetch_factor=4 if num_workers > 0 else None)

    all_feats = []
    all_labels = []
    n_done = 0
    t0 = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            feats = model(images)
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(np.array(labels, copy=True))
            n_done += images.size(0)
            del images, feats
            if n_done % (batch_size * 50) == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                logger.info("    extracted %d / %d (%.0f img/s)",
                            n_done, len(dataset), rate)

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    logger.info("    %s features: %s, %d samples, %.1f s",
                label, feats.shape, len(labels), time.time() - t0)
    return feats, labels


def _extract_features_local(model, dataset, device, batch_size, num_workers, logger, label=""):
    """Extract features from a local ImageFolder-style dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=False)
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images)
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(np.array(labels, copy=True))
            del images, feats

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    logger.info("    %s features: %s, %d samples", label, feats.shape, len(labels))
    return feats, labels


# Probe fitting and scoring (torch L-BFGS on GPU)
#
# Minimizes the identical objective as sklearn's LogisticRegression(solver="lbfgs"):
#   mean CE + 0.5 * ||W||^2 / (C * n)     (intercept unpenalized)
# under the identical stopping rule (tolerance_grad = PROBE_TOL unrescaled).
# float64 throughout, matching sklearn's arithmetic.
# See linear_probe_eval._fit_and_score_torch for the equivalence proof.

def _fit_probe_torch(feats, labels, C, max_iter, device, logger, label=""):
    """Fit W, b from zero via full-batch L-BFGS on GPU. Returns (W_np, b_np, classes, info).

    W_np is (d, k), b_np is (k,) — both float64 numpy on CPU. Kept beside
    `linear_probe_eval._fit_and_score_torch` (same objective and stopping rule) because
    this driver needs the fitted weights to score several OOD test sets.
    """
    import torch.nn.functional as F

    dev = torch.device(device)
    n, d = feats.shape
    classes, y_idx = np.unique(labels, return_inverse=True)
    k = len(classes)

    X = torch.as_tensor(np.asarray(feats, dtype=np.float64), device=dev)
    y = torch.as_tensor(y_idx, dtype=torch.long, device=dev)

    W = torch.zeros(d, k, dtype=torch.float64, device=dev, requires_grad=True)
    b = torch.zeros(k, dtype=torch.float64, device=dev, requires_grad=True)

    inv_cn = 1.0 / (C * n)
    opt = torch.optim.LBFGS(
        [W, b], max_iter=max_iter, max_eval=max_iter * 4, history_size=10,
        tolerance_grad=PROBE_TOL, tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )

    t0 = time.time()

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(X @ W + b, y) + 0.5 * inv_cn * (W * W).sum()
        loss.backward()
        return loss

    try:
        opt.step(closure)
        closure()
        grad_max = max(W.grad.abs().max().item(), b.grad.abs().max().item())
        Wn = W.detach().cpu().numpy()
        bn = b.detach().cpu().numpy()
    finally:
        del X, y, W, b
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    n_iter = int(opt.state_dict()["state"][0]["n_iter"])
    converged = grad_max <= PROBE_TOL
    elapsed = time.time() - t0

    logger.info("    %sprobe fit: C=%.3g, %d iters, converged=%s, %.1f s",
                label, C, n_iter, converged, elapsed)

    info = {
        "C": float(C),
        "n_iter": n_iter,
        "converged": converged,
        "solver": "torch",
        "n_classes": k,
    }
    return Wn, bn, classes, info


def _fit_and_score_val(x_fit, y_fit, x_val, y_val, logger, *, device, max_iter, C):
    """`linear_probe_eval._select_C`'s fit-and-score interface over `_fit_probe_torch`."""
    W, b, classes, info = _fit_probe_torch(x_fit, y_fit, C, max_iter, device, logger)
    pred = classes[np.argmax(np.asarray(x_val, dtype=np.float64) @ W + b, axis=1)]
    return float((pred == np.asarray(y_val)).mean()), info


def _select_C(train_feats, train_labels, device, logger):
    """C and the sweep record from the repo-wide selection (`linear_probe_eval._select_C`):
    the same grid, refinement, cold starts, iteration cap, and validation split as every
    other probe, so a record stamped with PROBE_C_SELECTION was selected that way."""
    return _select_C_shared(_fit_and_score_val, train_feats, train_labels, logger,
                            device=device)


def _score_topk(W, b, classes, test_feats, test_labels, class_mask=None):
    """Score test features through the fitted probe with optional class masking.

    W is (d, k), b is (k,) — both float64 numpy. Logits = X @ W + b.
    Returns (top1, top5). top5 is None when the class subset has <= 5 classes.
    """
    logits = np.asarray(test_feats, dtype=np.float64) @ W + b

    if class_mask is not None:
        n_classes = int(class_mask.sum())
        if n_classes < logits.shape[1]:
            mask_indices = np.where(class_mask)[0]
            logits = logits[:, mask_indices]
    else:
        n_classes = logits.shape[1]

    top1 = float((logits.argmax(axis=1) == test_labels).mean())

    k = min(5, n_classes)
    pred_topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    top5 = float((pred_topk == test_labels[:, None]).any(axis=1).mean())
    if n_classes <= 5:
        top5 = None

    return top1, top5


def discover_clip_runs(runs_root, logger, include_seed43=False):
    """Find all CLIP ViT-B/32 runs."""
    baseline_dir = runs_root / CLIP_BASELINE
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"CLIP baseline not found: {baseline_dir}")

    runs = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith(CLIP_PREFIX):
            continue
        is_seed43 = "seed43" in name
        if is_seed43 and not include_seed43:
            continue

        if name == CLIP_BASELINE or name == f"{CLIP_BASELINE}_seed43":
            onset = None
            is_baseline = True
        else:
            is_baseline = False
            try:
                onset = _onset_epoch(d)
            except Exception:
                logger.debug("  skipping %s (no onset epoch)", name)
                continue

        ckpt_dir = d / "checkpoints" / "all_epochs"
        if not ckpt_dir.is_dir():
            continue

        runs.append({
            "name": name,
            "dir": d,
            "onset": onset,
            "is_baseline": is_baseline,
        })

    return runs


CLIP_BASELINE_EPOCHS = [5, 6, 13, 14, 28, 29, 31]


def _select_epochs(run_info, avail):
    """Select onset, onset+1, final for perturbed; reference onsets + final for baseline."""
    avail_set = set(avail)
    if run_info["is_baseline"]:
        return sorted(e for e in CLIP_BASELINE_EPOCHS if e in avail_set)
    epochs = set()
    onset = run_info["onset"]
    if onset is not None and onset > 0:
        epochs.add(onset)
        epochs.add(onset + 1)
    if avail:
        epochs.add(avail[-1])
    return sorted(e for e in epochs if e in avail_set)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs_root", default=None, help="Default: paths key runs_root.")
    p.add_argument("--imagenet_root", default=None,
                   help="ImageNet with train/ and val/. Default: paths key imagenet_root.")
    p.add_argument("--runs", default="all", choices=["all", "baseline", "perturbed"])
    p.add_argument("--datasets", default=",".join(OOD_DATASETS))
    p.add_argument("--data_root", default=None, help="OOD datasets. Default: paths key probe_datasets.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--feature_cache", default=str(DEFAULT_FEATURE_CACHE))
    p.add_argument("--epochs", default=None,
                   help="Comma-separated epochs. Default: onset/onset+1/final per run.")
    p.add_argument("--clip_feature", default=DEFAULT_CLIP_FEATURE, choices=["cls", "projected"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-seed43", action="store_true",
                   help="Include seed-43 forks (skipped by default).")
    p.add_argument("--run_filter", default=None,
                   help="Keep only runs whose name contains this substring "
                        "(e.g. 'seed43' with --include-seed43 for the seed-43 family alone).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("clip_ood_probe")

    runs_root = Path(args.runs_root or paths.path("runs_root"))
    data_root = Path(args.data_root or paths.path("probe_datasets"))
    imagenet_root = Path(args.imagenet_root or paths.path("imagenet_root"))
    device = torch.device(args.device)
    ood_datasets = [d.strip() for d in args.datasets.split(",")]
    feature_cache = Path(args.feature_cache)

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    if args.clip_feature == "cls":
        output_dir = output_dir.parent / (output_dir.name + "_clip_cls")

    preprocessing = eval_preprocessing("CLIP ViT-B/32")
    transform = preprocessing.build(image_size=224)

    runs = discover_clip_runs(runs_root, logger, include_seed43=args.include_seed43)
    if args.runs == "baseline":
        runs = [r for r in runs if r["is_baseline"]]
    elif args.runs == "perturbed":
        runs = [r for r in runs if not r["is_baseline"]]
    if args.run_filter:
        runs = [r for r in runs if args.run_filter in r["name"]]

    logger.info("Found %d CLIP runs, feature=%s", len(runs), args.clip_feature)
    for r in runs:
        logger.info("  %s (onset=%s, baseline=%s)", r["name"], r["onset"], r["is_baseline"])

    total_completed = 0
    total_failed = 0

    for run in runs:
        run_name = run["name"]
        ckpt_dir = run["dir"] / "checkpoints" / "all_epochs"

        avail = sorted(e for e, _ in sorted_scratch_checkpoints(ckpt_dir))
        if args.epochs:
            epochs = [int(e) for e in args.epochs.split(",")]
        else:
            epochs = _select_epochs(run, avail)

        if not epochs:
            logger.info("  [skip] %s -- no matching epochs", run_name)
            continue

        logger.info("\n%s", "=" * 72)
        logger.info("%s -- epochs %s", run_name, epochs)
        logger.info("=" * 72)

        if args.dry_run:
            for ep in epochs:
                logger.info("  [dry-run] %s @ epoch %d", run_name, ep)
                total_completed += 1
            continue

        for ep in epochs:
            out_run_dir = output_dir / run_name
            out_run_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_run_dir / f"ood_probe_eval_epoch{ep}.json"

            if out_file.exists() and not args.force:
                try:
                    cached = json.load(open(out_file))
                    if is_current_recipe(cached) and "imagenet_val" in cached:
                        logger.info("  [skip] %s @ %d (cached)", run_name, ep)
                        total_completed += 1
                        continue
                except Exception:
                    pass

            logger.info("\n--- %s @ epoch %d ---", run_name, ep)
            t_ckpt = time.time()

            try:
                ckpt_map = dict(sorted_scratch_checkpoints(ckpt_dir))
                ckpt_path = ckpt_map.get(ep)
                if ckpt_path is None:
                    logger.warning("  [skip] no checkpoint for epoch %d", ep)
                    continue

                model = load_clip_feature_model(
                    str(ckpt_path), clip_feature=args.clip_feature, device=str(device),
                )
                model.eval()

                feat_cache_dir = feature_cache / f"{run_name}_ood_ep{ep}"
                feat_cache_dir.mkdir(parents=True, exist_ok=True)
                train_feat_path = feat_cache_dir / "imagenet_train.npz"

                if train_feat_path.exists() and not args.force:
                    logger.info("  loading cached ImageNet train features")
                    data = np.load(train_feat_path)
                    train_feats, train_labels = data["feats"], data["labels"]
                    logger.info("  ImageNet train: %s", train_feats.shape)
                else:
                    logger.info("  extracting ImageNet-1k train features...")
                    train_feats, train_labels = _extract_features_imagenet(
                        model, imagenet_root / "train", transform, device,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        logger=logger, label="ImageNet train",
                    )
                    np.savez_compressed(train_feat_path,
                                        feats=train_feats.astype(np.float32),
                                        labels=train_labels)
                    logger.info("  cached to %s (%.1f MB)",
                                train_feat_path,
                                train_feat_path.stat().st_size / 1e6)

                n_features = train_feats.shape[1]
                const_mask = train_feats.std(axis=0) == 0
                n_const = int(const_mask.sum())
                if n_const > 0:
                    keep = ~const_mask
                    train_feats = train_feats[:, keep]
                    logger.info("  dropped %d/%d constant dims; %d remain",
                                n_const, n_features, n_features - n_const)
                standardize = fit_standardizer(train_feats)
                train_feats = standardize(train_feats)

                best_c, sweep_info = _select_C(train_feats, train_labels, device, logger)

                logger.info("  fitting final probe on full ImageNet train...")
                W, b_vec, classes, fit_info = _fit_probe_torch(
                    train_feats, train_labels, best_c,
                    PROBE_FINAL_MAX_ITER, device, logger,
                )

                if not fit_info["converged"]:
                    logger.warning("  [not-converged] ImageNet probe stopped at %d iters",
                                   fit_info["n_iter"])

                del train_feats, train_labels
                gc.collect()

                logger.info("  extracting ImageNet-1k val features...")
                val_feats, val_labels = _extract_features_imagenet(
                    model, imagenet_root / "val", transform, device,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    logger=logger, label="ImageNet val",
                )
                if n_const > 0:
                    val_feats = val_feats[:, keep]
                val_feats = standardize(val_feats)

                imagenet_top1, imagenet_top5 = _score_topk(W, b_vec, classes,
                                                              val_feats, val_labels)
                n_val_actual = len(val_labels)
                logger.info("  ImageNet-1k val: top-1 = %.4f  top-5 = %.4f  (%d images)",
                            imagenet_top1, imagenet_top5, n_val_actual)

                del val_feats, val_labels
                gc.collect()

                ood_results = {}
                for ds_name in ood_datasets:
                    logger.info("  scoring %s ...", ds_name)
                    try:
                        ds, class_mask = _build_ood_dataset(
                            ds_name, transform, data_root, download=False,
                        )
                        test_feats, test_labels = _extract_features_local(
                            model, ds, device, args.batch_size, args.num_workers,
                            logger, label=ds_name,
                        )
                        if n_const > 0:
                            test_feats = test_feats[:, keep]
                        test_feats = standardize(test_feats)

                        top1, top5 = _score_topk(W, b_vec, classes,
                                                    test_feats, test_labels, class_mask)

                        n_classes = int(class_mask.sum())
                        logger.info("    %s: top-1 = %.4f  top-5 = %s  (%d classes, %d images)",
                                    ds_name, top1,
                                    f"{top5:.4f}" if top5 is not None else "n/a",
                                    n_classes, len(test_labels))

                        ood_results[ds_name] = {
                            "top1_accuracy": top1,
                            "top5_accuracy": top5,
                            "n_classes": n_classes,
                            "n_test": len(test_labels),
                        }
                        del test_feats, test_labels

                    except Exception as exc:
                        logger.exception("    [fail] %s: %s", ds_name, exc)
                        ood_results[ds_name] = {"error": f"{type(exc).__name__}: {exc}"}

                record = {
                    "architecture": "CLIP ViT-B/32",
                    "run_name": run_name,
                    "is_baseline": run["is_baseline"],
                    "perturb_type": None if run["is_baseline"] else _parse_perturb_type(run_name),
                    "phase": None if run["is_baseline"] else _parse_phase(run_name),
                    "epoch": ep,
                    "checkpoint": ckpt_path.name,
                    "clip_feature": args.clip_feature,
                    "evaluation_type": "ood_linear_probe",
                    "train_dataset": "imagenet1k_train",
                    "n_train": IMAGENET_TRAIN_SIZE,
                    "n_features": n_features,
                    "n_constant_dims_dropped": n_const,
                    "preprocessing": preprocessing.tag,
                    "c_selection": PROBE_C_SELECTION,
                    "feature_scaling": PROBE_FEATURE_SCALING,
                    "c_sweep": sweep_info,
                    **fit_info,
                    "imagenet_val": {
                        "top1_accuracy": imagenet_top1,
                        "top5_accuracy": imagenet_top5,
                        "n_test": n_val_actual,
                    },
                    "results": ood_results,
                }

                with open(out_file, "w") as f:
                    json.dump(record, f, indent=2)
                logger.info("  saved %s", out_file)
                total_completed += 1

                del W, b_vec, classes, model
                gc.collect()
                torch.cuda.empty_cache()

                import shutil
                if feat_cache_dir.exists():
                    sz = sum(f.stat().st_size for f in feat_cache_dir.rglob("*") if f.is_file())
                    shutil.rmtree(feat_cache_dir)
                    logger.info("  [cleanup] %s (%.0f MB freed)", feat_cache_dir, sz / 1e6)

            except Exception as exc:
                logger.exception("  [FAILED] %s @ %d: %s", run_name, ep, exc)
                total_failed += 1

            elapsed = time.time() - t_ckpt
            logger.info("  checkpoint done in %.1f min", elapsed / 60)

    logger.info("\n" + "=" * 60)
    logger.info("CLIP OOD probe finished. Completed: %d  Failed: %d", total_completed, total_failed)
    logger.info("=" * 60)


def _parse_perturb_type(run_name):
    for pt in PERTURB_TYPES:
        if pt in run_name:
            return pt
    return None


def _parse_phase(run_name):
    for phase in PHASES:
        if run_name.endswith(f"_{phase}"):
            return phase
    return None


if __name__ == "__main__":
    main()
