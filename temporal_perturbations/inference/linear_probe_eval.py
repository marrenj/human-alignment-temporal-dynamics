"""Linear-probe evaluation for ResNet-50, ViT-B/16, ConvNeXt-Base, and CLIP-ViT-B/32 models on the 29 classification datasets
from the LAION clip_benchmarks package.

For each dataset:
  1. Extract frozen features from the train + test splits using the model's
     penultimate layer (timm's `forward_features` + global pooling for ResNet-50, ViT-B/16, ConvNeXt-Base) 
     or the vision encoder's CLS token for CLIP-ViT-B/32.
  2. Fit multinomial logistic regression (L-BFGS), selecting the L2 strength C on a
     held-out validation slice of the training set on a coarse-to-fine grid (`_select_C`),
     then refit on the full training set at the chosen C.
  3. Report top-1 and top-5 accuracy on the test set.

Features are cached to <output_dir>/features/<dataset>_<split>.npy keyed on the
checkpoint epoch so probes can be re-fit cheaply if the regression hyperparameters need tweaking.
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from temporal_perturbations.data.eval_transforms import EvalPreprocessing


# 29 classification datasets (31 datasets minus the 2 retrieval ones).
#
# The `vtab/` entries use VTAB's canonical splits, which is what clip_benchmark's own
# collections use and what published tables report. The bare names are not usable:
# four of them (svhn, clevr_*, kitti_*) do not exist unprefixed, and two (caltech101,
# eurosat) silently return the SAME data for train and test because their plain
# builders ignore `split`.
#
# vtab/* datasets are read from PNG trees produced by
# pipeline/scripts/downloads/export_vtab_datasets.py (separate conda env -- VTAB's split
# definitions need TensorFlow). A vtab/ name with no export raises FileNotFoundError.
PAPER_PROBE_DATASETS = [
    "imagenet1k",
    "imagenet-a",
    "imagenet-o",
    "imagenet-r",
    "imagenet_sketch",
    "imagenetv2",
    "objectnet",
    "vtab/cifar100",
    "vtab/cifar10",
    "vtab/caltech101",
    "vtab/clevr_closest_object_distance",
    "vtab/clevr_count_all",
    "vtab/kitti_closest_vehicle_distance",
    "vtab/dtd",
    "vtab/eurosat",
    "vtab/flowers",
    "vtab/pets",
    "vtab/resisc45",
    "vtab/svhn",
    "vtab/pcam",
    "vtab/cars",
    "vtab/food101",
    "fgvc_aircraft",
    "voc2007",
    "country211",
    "gtsrb",
    "mnist",
    "renderedsst2",
    "stl10",
]


def _extract_features(model: nn.Module, loader: DataLoader, device: torch.device, logger: logging.Logger, split: str = ""):
    """
    Passes dataset through frozen backbone once to pre-extract feature tensors.
    """
    feats, labels = [], []
    n_batches = len(loader)
    step = max(1, n_batches // 10)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                imgs, lbls = batch[0], batch[1]
            elif isinstance(batch, dict):
                imgs = batch["image"] if "image" in batch else batch["images"]
                lbls = batch["label"] if "label" in batch else batch["labels"]
            else:
                raise RuntimeError(f"Unrecognized batch shape: {type(batch)}")
            del batch

            imgs = imgs.to(device, non_blocking=True)

            if hasattr(model, "forward_features"):
                raise TypeError(
                    f"Raw timm model passed to _extract_features ({type(model).__name__}). "
                    "Wrap it in TimmFeatureWrapper first — hand-rolled pooling "
                    "skips ConvNeXt's post-pool LayerNorm."
                )
            f = model(imgs)

            feats.append(f.detach().cpu().float().numpy())
            labels.append(np.array(lbls, copy=True))

            if batch_idx % step == 0 or batch_idx == n_batches - 1:
                seen = len(feats) * imgs.shape[0]
                dt = time.time() - t0
                rate = seen / dt if dt > 0 else 0.0
                eta = (n_batches - batch_idx - 1) * (dt / (batch_idx + 1))
                logger.info("    extract%s: %d/%d batches (%.0f%%) ~%.0f img/s, eta %.0fs",
                            f" {split}" if split else "", batch_idx + 1, n_batches,
                            100.0 * (batch_idx + 1) / n_batches, rate, eta)

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


# Datasets built directly from torchvision rather than through clip_benchmark.
# These are the ones the generalization sweep actually uses, and torchvision gives
# explicit control over the download location and the train/test split, whereas
# clip_benchmark's builder picks its own root and download policy.
_TORCHVISION_BUILDERS = {"cifar10": "CIFAR10", "cifar100": "CIFAR100"}

# `vtab/*` datasets are pre-materialized to ImageFolder trees under this subdirectory of
# data_root by pipeline/scripts/downloads/export_vtab_datasets.py, which runs in a separate
# conda env because VTAB's split definitions need TensorFlow (numpy 2.x) and this env pins
# numpy 1.26. See that script's docstring.
_VTAB_EXPORT_SUBDIR = "vtab_export"

# torchvision's Country211 hardcodes a `root / "country211"` subdirectory; the full
# archive is extracted under this wrapper directory (symlink country211 -> country211_full).
_COUNTRY211_SUBDIR = "country211_full_root"


def _vtab_split_dataset(export_dir, split, transform):
    """One split of an exported VTAB dataset, with labels decoded from directory names.

    ImageFolder assigns labels by position in the sorted list of directories PRESENT in
    that split, so if train and test contain different class subsets their label spaces
    silently disagree -- a probe would then fit on one labelling and score against
    another with no error raised. The directory name IS the VTAB label, so decode it
    rather than trusting positional order.
    """
    from torchvision.datasets import ImageFolder

    split_dir = Path(export_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"No VTAB export at {split_dir}. Run export_vtab_datasets.py with the "
            f"vtab-prep interpreter first."
        )
    ds = ImageFolder(str(split_dir), transform=transform)
    idx_to_label = {idx: int(name) for name, idx in ds.class_to_idx.items()}
    ds.target_transform = idx_to_label.__getitem__
    return ds


def _stratified_subset(dataset, max_samples, seed=42):
    """Stratified subsample: equal samples per class, remainder allocated round-robin."""
    targets = np.array([dataset.targets[i] for i in range(len(dataset))])
    classes = np.unique(targets)
    per_class = max_samples // len(classes)
    remainder = max_samples - per_class * len(classes)

    rng = np.random.RandomState(seed)
    indices = []
    for i, c in enumerate(classes):
        class_idx = np.where(targets == c)[0]
        n = per_class + (1 if i < remainder else 0)
        n = min(n, len(class_idx))
        indices.append(rng.choice(class_idx, size=n, replace=False))
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return torch.utils.data.Subset(dataset, indices.tolist())


def _build_dataset_loaders(dataset_name, transform, batch_size, num_workers,
                           data_root=None, download=True, max_train_samples=None):
    """Build (train_loader, test_loader) for `dataset_name`.

    CIFAR-10/100 come straight from torchvision (see `_TORCHVISION_BUILDERS`);
    anything else falls back to clip_benchmark's registry, which covers the rest
    of the 27-dataset paper suite.
    """
    if data_root is None:
        raise ValueError(f"{dataset_name} needs a data_root to download/read into.")
    if dataset_name == "country211":
        data_root = Path(data_root) / _COUNTRY211_SUBDIR
    Path(data_root).mkdir(parents=True, exist_ok=True)

    if dataset_name in ("imagenet1k", "imagenet1k_16superclass"):
        from torchvision.datasets import ImageFolder

        from temporal_perturbations import paths

        imagenet_root = paths.path("imagenet_root")
        train_ds = ImageFolder(str(imagenet_root / "train"), transform=transform)
        test_ds = ImageFolder(str(imagenet_root / "val"), transform=transform)

        if dataset_name == "imagenet1k_16superclass":
            from temporal_perturbations.data.imagenet_superclasses import make_superclass_target_transform
            tt = make_superclass_target_transform(train_ds.class_to_idx)
            train_ds.target_transform = tt
            test_ds.target_transform = tt

        if max_train_samples is not None and max_train_samples < len(train_ds):
            train_ds = _stratified_subset(train_ds, max_train_samples)
    elif dataset_name.startswith("vtab/"):
        export_dir = Path(data_root) / _VTAB_EXPORT_SUBDIR / dataset_name[len("vtab/"):]
        train_ds = _vtab_split_dataset(export_dir, "train", transform)
        test_ds = _vtab_split_dataset(export_dir, "test", transform)
    elif dataset_name in _TORCHVISION_BUILDERS:
        from torchvision import datasets as tv_datasets

        cls = getattr(tv_datasets, _TORCHVISION_BUILDERS[dataset_name])
        train_ds = cls(root=str(data_root), train=True, transform=transform, download=download)
        test_ds = cls(root=str(data_root), train=False, transform=transform, download=download)
    else:
        from clip_benchmark.datasets.builder import build_dataset

        # `root` must be passed explicitly: it defaults to the RELATIVE string "root",
        # so every dataset would land in a `root/` dir under the cwd. Threading data_root/download through makes --data_root and
        # --no_download mean the same thing for all 29 datasets, not just the two
        # torchvision ones.
        train_ds = build_dataset(dataset_name=dataset_name, transform=transform,
                                 split="train", root=str(data_root), download=download)
        test_ds = build_dataset(dataset_name=dataset_name, transform=transform,
                                split="test", root=str(data_root), download=download)

    common = dict(batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    return (
        torch.utils.data.DataLoader(train_ds, **common),
        torch.utils.data.DataLoader(test_ds, **common),
    )


# gzip on float32 activations only pays when the data is sparse, and whether it is
# depends on the CHECKPOINT, not the dataset -- so the choice is made per array rather
# than globally. Measured at PCam's 262144x2048:
#   dense baseline features   (32% zeros): 29.5% smaller for 40x  the write time -- bad
#   degenerate perturbed feats(87% zeros): 86.6% smaller for 8.6x the write time -- good
# Perturbed onset checkpoints are the degenerate case (1468/2048
# constant units, 87% exact zeros for RN50 image_noise), which is where a 36-run sweep
# goes from ~129 GB to ~18 GB. Compression is LOSSLESS either way -- DEFLATE round-trips
# float32 bit-exactly -- so this is purely a disk/time tradeoff, never an accuracy one.
_COMPRESS_ZERO_FRACTION = 0.5


def _save_features(path, feats, labels, logger):
    """Write a feature cache, compressing only when the array is sparse enough to benefit."""
    zero_frac = float((feats == 0).mean())
    compress = zero_frac >= _COMPRESS_ZERO_FRACTION
    (np.savez_compressed if compress else np.savez)(path, feats=feats, labels=labels)
    logger.info("    cached %s (%.0f%% zeros, %s)", Path(path).name, 100 * zero_frac,
                "compressed" if compress else "uncompressed")
    return compress


PROBE_C = 1.0
PROBE_TOL = 1e-4
PROBE_TOP_K = 5

# Kornblith/Radford C selection: the CLIP paper's coarse parametric-search grid
# (7 points, 100x apart), then PROBE_C_REFINE_ROUNDS rounds of halving the
# log-interval around the peak (final resolution ~3.2x), scored on a stratified
# validation slice of train; the winner is refit on the FULL train split. Sweep fits
# are capped at PROBE_SWEEP_MAX_ITER -- selection needs a ranking, not convergence --
# so the weakly-regularized fits on a degenerate checkpoint stay bounded; only the
# final fit gets the full iteration budget.
PROBE_C_GRID = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6)
PROBE_C_REFINE_ROUNDS = 2
PROBE_VAL_FRACTION = 0.2
PROBE_VAL_SEED = 0
PROBE_SWEEP_MAX_ITER = 2000
PROBE_C_SELECTION = "val_sweep_v2"  # recipe marker (cold-start sweep); every cache gate keys on it

# Every feature is z-scored with training-split statistics before any fit. The stopping
# rule is an absolute gradient tolerance (PROBE_TOL, scikit-learn's), and at zero weights
# the gradient for class c is (overall mean - class-c mean) / k in RAW feature units. On a
# perturbed checkpoint the class-dependent differences sit in low-variance directions, so
# L-BFGS declared convergence at or near zero weights: ConvNeXt-B's 1000-class ImageNet
# probe after a late label-shuffle epoch started with max |grad| = 9.5e-6, stopped at
# iteration 0 and scored 0.1%; standardized, it fits to 36.5%. Healthy baselines move by
# -0.1 to +1.4 pp. The floor keeps numerically constant units (std at float32 noise) from
# being blown up to unit scale.
PROBE_FEATURE_SCALING = "standardized"  # recorded in every block; gated with PROBE_C_SELECTION
PROBE_STD_FLOOR = 1e-3  # relative to the RMS of the per-feature std


def fit_standardizer(train_feats):
    """Return a function that z-scores features with `train_feats`' mean and std."""
    mu = train_feats.mean(axis=0, dtype=np.float64)
    sd = train_feats.std(axis=0, dtype=np.float64)
    sd = np.maximum(sd, PROBE_STD_FLOOR * np.sqrt(np.mean(sd ** 2)))

    # Applied in the input's dtype: for CLIP's 1.28M x 768 ImageNet features a float64
    # intermediate would cost ~8 GB, and z-scored values need no more than float32.
    def standardize(feats):
        dtype = np.result_type(feats.dtype, np.float32)  # integer inputs (pixels) -> float
        return (feats.astype(dtype, copy=False) - mu.astype(dtype)) / sd.astype(dtype)

    return standardize


def is_current_recipe(record):
    """True when a probe record was fit under the current C selection and feature scaling."""
    return (record.get("c_selection") == PROBE_C_SELECTION
            and record.get("feature_scaling") == PROBE_FEATURE_SCALING)


def _penalized_nll(W, b, feats, y_idx, C):
    """The value scikit-learn's lbfgs minimizes, for cross-solver comparison.

        0.5 * ||W||^2  +  C * sum_i CE(W x_i + b, y_i)      (intercept unpenalized)

    Reported alongside every fit so a torch result and a sklearn result on the same
    features can be checked against each other directly: same objective, same
    problem, so the lower value is the better-optimized fit.
    """
    logits = feats @ W + b
    logits = logits - logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(logits).sum(axis=1))
    nll = -(logits[np.arange(len(y_idx)), y_idx] - log_norm).sum()
    return float(0.5 * (W ** 2).sum() + C * nll)


def _top_k_accuracy(scores, classes, test_labels, k=PROBE_TOP_K):
    """Fraction of test samples whose true label is among the k highest-scoring classes.

    `None` rather than 1.0 when the dataset has k or fewer classes: the metric is
    correct-by-construction there, and a column of 1.0s next to real top-5 numbers
    reads as a result instead of as N/A. PAPER_PROBE_DATASETS holds two binary
    datasets (`pcam`, `renderedsst2`), so this is a live case.

    argpartition rather than a sort -- only membership in the top k matters, not the
    order within it.
    """
    if len(classes) <= k:
        return None
    topk = classes[np.argpartition(scores, -k, axis=1)[:, -k:]]
    return float((topk == np.asarray(test_labels)[:, None]).any(axis=1).mean())


def _fit_and_score(train_feats, train_labels, test_feats, test_labels, logger,
                   *, max_iter=1000, C=PROBE_C, **_unused):
    """Fit logistic regression on train features and report top-1/top-5 on test (CPU)."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    # `multi_class` is deliberately not passed: "auto" was already the default
    # behaviour (multinomial for lbfgs with >2 classes), the argument was
    # deprecated in scikit-learn 1.5 and removed in 1.7, so passing it buys
    # nothing and breaks the eval outright on a current scikit-learn.
    #
    # `n_jobs` is likewise not passed: it only ever parallelized the one-vs-rest
    # outer loop, which multinomial lbfgs does not have, so it was a no-op here
    # even before scikit-learn 1.8 deprecated it. Threading comes from the BLAS
    # underneath (OMP_NUM_THREADS / MKL_NUM_THREADS), not from this argument.
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        tol=PROBE_TOL,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(train_feats, train_labels)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    # `predict_proba`, not `decision_function`: the latter collapses to a single column
    # for k=2, which `_top_k_accuracy`'s row-wise argpartition cannot consume. Ranking is
    # unchanged either way (softmax for k>2, expit for k=2, both strictly monotone), so
    # top-1 here is identical to `clf.predict`.
    scores = clf.predict_proba(test_feats)
    acc = float((clf.classes_[scores.argmax(axis=1)] == np.asarray(test_labels)).mean())
    top5 = _top_k_accuracy(scores, clf.classes_, test_labels)
    _, y_idx = np.unique(train_labels, return_inverse=True)
    info = {
        "solver": "sklearn",
        "C": float(C),
        "converged": converged,
        "n_iter": int(np.max(clf.n_iter_)),
        "objective": _penalized_nll(clf.coef_.T.astype(np.float64),
                                    clf.intercept_.astype(np.float64),
                                    train_feats.astype(np.float64), y_idx, C),
        "top5_accuracy": top5,
        "n_classes": int(len(clf.classes_)),
    }
    logger.info("    top-1 = %.4f  top-5 = %s  (sklearn, C=%.3g, %d iters, converged=%s)",
                acc, "n/a" if top5 is None else f"{top5:.4f}", C, info["n_iter"], converged)
    return acc, info


def _fit_and_score_torch(train_feats, train_labels, test_feats, test_labels, logger,
                         *, device, max_iter=20000, C=PROBE_C,
                         W_init=None, b_init=None, **_unused):
    """Same fit as `_fit_and_score`, but the linear algebra runs on `device`.

    The fit is on `device`; the final scoring matmul is deliberately left on the CPU.
    `Wn`/`bn` are copied off the device before the `finally` frees the float64 feature
    tensors, and a 10k x d @ d x k product costs milliseconds -- keeping the test
    features resident to score on GPU would re-open the OOM path guarded below for no
    measurable gain.

    Minimizes the identical objective (`_penalized_nll`, intercept unpenalized) with
    full-batch L-BFGS under the identical stopping rule -- scikit-learn hands scipy
    `gtol=tol` over the *mean*-scaled objective (`_linear_loss` divides the loss by
    the sample-weight sum and sets `l2_reg_strength = 1/(C*n)`), which is exactly the
    `closure` below, so `tolerance_grad` is PROBE_TOL unrescaled. Same problem, same
    stopping rule, same answer -- only the hardware changes. The probe is one big
    matmul per iteration, which is what a GPU is for; the CPU fit took 1-4 h per
    dataset.

    float64 throughout, matching scikit-learn's own arithmetic. float32 would clear
    the 1e-4 gradient tolerance comfortably and run ~64x faster on a consumer
    NVIDIA GPU, but these features are extremely ill-conditioned -- a perturbed
    checkpoint's Gram matrix has a condition number around 1e34 -- and fp32 error
    there is not worth trading against a paper number. If probe time ever becomes the
    binding constraint, fp32 is the lever to pull, but TF32 matmuls must be disabled
    with it (~1e-3 relative error, far above the tolerance).

    One deliberate difference, for the two binary datasets in PAPER_PROBE_DATASETS
    (`pcam`, `renderedsst2`): scikit-learn fits a single logit vector for k=2, while
    this always fits the k-class softmax. Same decision-boundary family, but the L2
    penalty sees a redundant parameterization, so those two are not bit-comparable
    across solvers. Irrelevant to the 10/100-class CIFAR probes.

    When `W_init`/`b_init` are provided (numpy arrays from a previous fit), they
    warm-start the optimizer -- Kornblith et al. 2019's regularization-path trick.
    The objective is convex, so the solution is identical regardless of initialization;
    warm-starting just converges faster because neighboring C values have nearby optima.
    """
    import torch.nn.functional as F

    dev = torch.device(device)
    n, d = train_feats.shape
    classes, y_idx = np.unique(train_labels, return_inverse=True)

    X = torch.as_tensor(np.asarray(train_feats, dtype=np.float64), device=dev)
    y = torch.as_tensor(y_idx, dtype=torch.long, device=dev)

    if W_init is not None:
        W = torch.as_tensor(W_init.copy(), device=dev, dtype=torch.float64)
        W.requires_grad_(True)
    else:
        W = torch.zeros(d, len(classes), dtype=torch.float64, device=dev, requires_grad=True)

    if b_init is not None:
        b = torch.as_tensor(b_init.copy(), device=dev, dtype=torch.float64)
        b.requires_grad_(True)
    else:
        b = torch.zeros(len(classes), dtype=torch.float64, device=dev, requires_grad=True)

    # Mean cross-entropy plus 0.5*||W||^2/(C*n) -- byte-for-byte the objective
    # sklearn hands scipy, so PROBE_TOL applies to this gradient directly with no
    # rescaling. `max_eval` must be raised, though: torch defaults it to
    # max_iter*1.25, but strong_wolfe spends 2-3 closure evals per iteration, so the
    # default silently halts the fit at roughly half the requested iterations.
    inv_cn = 1.0 / (C * n)
    opt = torch.optim.LBFGS(
        [W, b], max_iter=max_iter, max_eval=max_iter * 4, history_size=10,
        tolerance_grad=PROBE_TOL, tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(X @ W + b, y) + 0.5 * inv_cn * (W * W).sum()
        loss.backward()
        return loss

    try:
        opt.step(closure)
        closure()  # repopulate .grad at the final iterate for the convergence check
        grad_max = max(W.grad.abs().max().item(), b.grad.abs().max().item())
        Wn = W.detach().cpu().numpy()
        bn = b.detach().cpu().numpy()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"Probe fit ran out of memory on {dev}: float64 features for this dataset "
            f"need ~{n * d * 8 / 2**30:.1f} GiB resident. Re-run with "
            f"--probe_solver sklearn to fit on CPU instead."
        ) from exc
    finally:
        del X, y, W, b
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # Bound once: both metrics read the same score matrix, and recomputing it inside a
    # second argmax would repeat the (n_test x d) @ (d x k) product.
    scores = np.asarray(test_feats, dtype=np.float64) @ Wn + bn
    acc = float((classes[np.argmax(scores, axis=1)] == np.asarray(test_labels)).mean())
    top5 = _top_k_accuracy(scores, classes, test_labels)
    info = {
        "solver": "torch",
        "C": float(C),
        "converged": grad_max <= PROBE_TOL,
        "n_iter": int(opt.state_dict()["state"][0]["n_iter"]),
        "objective": _penalized_nll(Wn, bn, np.asarray(train_feats, dtype=np.float64),
                                    y_idx, C),
        "top5_accuracy": top5,
        "n_classes": int(len(classes)),
        "W": Wn,
        "b": bn,
    }
    logger.info("    top-1 = %.4f  top-5 = %s  (torch/%s, C=%.3g, %d iters, converged=%s)",
                acc, "n/a" if top5 is None else f"{top5:.4f}", dev, C,
                info["n_iter"], info["converged"])
    return acc, info


PROBE_SOLVERS = {"torch": _fit_and_score_torch, "sklearn": _fit_and_score}
DEFAULT_PROBE_SOLVER = "torch"


def _val_split(train_feats, train_labels):
    """Stratified PROBE_VAL_FRACTION slice of train for C selection, fixed seed.

    Singleton classes (one sample) are pulled into the fit slice before
    splitting -- they can't be split across two sets, and one sample
    contributes nothing to a val accuracy ranking.
    """
    from sklearn.model_selection import train_test_split

    unique, counts = np.unique(train_labels, return_counts=True)
    singleton_classes = set(unique[counts < 2])

    if not singleton_classes:
        return train_test_split(train_feats, train_labels, test_size=PROBE_VAL_FRACTION,
                                random_state=PROBE_VAL_SEED, stratify=train_labels)

    mask = np.array([lb not in singleton_classes for lb in train_labels])
    x_fit, x_val, y_fit, y_val = train_test_split(
        train_feats[mask], train_labels[mask], test_size=PROBE_VAL_FRACTION,
        random_state=PROBE_VAL_SEED, stratify=train_labels[mask],
    )
    x_fit = np.concatenate([x_fit, train_feats[~mask]])
    y_fit = np.concatenate([y_fit, train_labels[~mask]])
    return x_fit, x_val, y_fit, y_val


def _select_C(fit_and_score, train_feats, train_labels, logger, *, device):
    """Pick the L2 strength C by cold-starting each candidate independently.

    Every C in the coarse grid starts from zero weights, so no candidate gains an
    advantage from a warm start, and the absolute gradient tolerance cannot end a
    fit before its first step (warm-started from a neighbor's solution, L-BFGS
    returns at iteration 0 whenever |W*/(C_prev * n)| < tol). Then
    PROBE_C_REFINE_ROUNDS of interval-halving around the peak. Ties break toward
    the SMALLER C (stronger regularization): on degenerate features several C
    values often reach the same val accuracy, and the more regularized fit is the
    one that converges rather than wandering an ill-conditioned plateau.

    The inner fits' own top-1 log lines are demoted below INFO -- they are val
    scores and would read as test results in the log -- and replaced with one
    summary line per candidate C.
    """
    x_fit, x_val, y_fit, y_val = _val_split(train_feats, train_labels)
    quiet = logger.getChild("sweep")
    quiet.setLevel(logging.WARNING)

    scores = {}

    def evaluate(log_c):
        if log_c in scores:
            return
        c = 10.0 ** log_c
        acc, info = fit_and_score(x_fit, y_fit, x_val, y_val, quiet,
                                  device=device, max_iter=PROBE_SWEEP_MAX_ITER, C=c)
        scores[log_c] = acc
        info.pop("W", None)
        info.pop("b", None)
        logger.info("    sweep C=%.3g: val top-1 = %.4f (%d iters%s)",
                    c, acc, info["n_iter"], "" if info["converged"] else ", capped")

    sorted_grid = sorted(math.log10(c) for c in PROBE_C_GRID)
    for log_c in sorted_grid:
        evaluate(log_c)

    for _ in range(PROBE_C_REFINE_ROUNDS):
        grid = sorted(scores)
        peak = min(grid, key=lambda g: (-scores[g], g))
        i = grid.index(peak)
        if i > 0:
            mid = (grid[i - 1] + peak) / 2
            evaluate(mid)
        if i < len(grid) - 1:
            mid = (peak + grid[i + 1]) / 2
            evaluate(mid)

    grid = sorted(scores)
    best = min(grid, key=lambda g: (-scores[g], g))
    logger.info("    selected C=%.3g (val top-1 = %.4f over %d candidates)",
                10.0 ** best, scores[best], len(grid))
    return 10.0 ** best, {
        "C_grid": [10.0 ** g for g in grid],
        "val_top1": [scores[g] for g in grid],
        "n_val": int(len(y_val)),
        "val_fraction": PROBE_VAL_FRACTION,
        "val_seed": PROBE_VAL_SEED,
        "sweep_max_iter": PROBE_SWEEP_MAX_ITER,
    }


def run_linear_probe(
    model,
    *,
    output_dir: str | Path,
    epoch: str | int,
    preprocessing: EvalPreprocessing,
    datasets: list[str] | None = None,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 8,
    device: str | torch.device = "cuda",
    force: bool = False,
    data_root: str | Path | None = None,
    download: bool = True,
    feature_cache_dir: str | Path | None = None,
    probe_solver: str = DEFAULT_PROBE_SOLVER,
    probe_max_iter: int | None = None,
    max_train_samples: int | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Run linear-probe eval on `model` over `datasets`.

    `preprocessing` is required and has no default: it must be the recipe `model`'s
    checkpoint was trained under (`data.eval_transforms.eval_preprocessing`), because
    feeding a model the wrong normalization or crop geometry silently costs accuracy
    rather than failing. Its `tag` keys the caches, so the same checkpoint evaluated
    under two recipes cannot collide.

    `probe_solver` picks where the logistic regression runs -- "torch" (default,
    full-batch L-BFGS on `device`) or "sklearn" (CPU reference). Both minimize the
    same objective under the same stopping rule; see `_fit_and_score_torch`.

    The L2 strength C is selected per (dataset, checkpoint) on a held-out validation
    slice of train (`_select_C`, the Radford/Kornblith protocol), then the probe is
    refit on the full train split at the winner. `probe_max_iter` caps only that
    final fit; selection fits are always capped at PROBE_SWEEP_MAX_ITER. Features are
    z-scored with training-split statistics first (`fit_standardizer`; see
    PROBE_FEATURE_SCALING for why). Each result block records the chosen `C`, the sweep
    curve (`c_sweep`), and the recipe markers `c_selection` and `feature_scaling` -- the
    cache gate (`is_current_recipe`) rejects records without them, so results from an
    older recipe re-fit from cached features.

    Returns
    -------
    dict
        ``{dataset: {"epoch": ..., "evaluation_type": "linear_probe",
                     "dataset": ..., "score": top1_accuracy,
                     "top5_accuracy": ..., "n_classes": ..., ...}}``

        ``score`` stays top-1 -- it is the headline every downstream reader already
        keys on. ``top5_accuracy`` is ``None`` for datasets with at most
        ``PROBE_TOP_K`` classes, where the metric is correct by construction.
    """
    log = logger or logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extracted features are regenerable intermediates and are written/read in bulk,
    # so callers keep them out of the results tree (typically under feature_cache).
    feat_dir = Path(feature_cache_dir) if feature_cache_dir else (output_dir / "features")
    feat_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = list(PAPER_PROBE_DATASETS)

    if probe_solver not in PROBE_SOLVERS:
        raise ValueError(f"probe_solver must be one of {sorted(PROBE_SOLVERS)}, "
                         f"got {probe_solver!r}")
    fit_and_score = PROBE_SOLVERS[probe_solver]
    fit_kwargs = {} if probe_max_iter is None else {"max_iter": probe_max_iter}

    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device).eval()

    transform = preprocessing.build(image_size=image_size)
    log.info("  [preprocessing] %s: resize %d -> crop %d (%s), mean=%s std=%s",
             preprocessing.tag, round(image_size * preprocessing.resize_ratio),
             image_size, preprocessing.interpolation,
             preprocessing.mean, preprocessing.std)

    aggregated = {}
    successes, failures = [], []

    for ds_name in datasets:
        safe = ds_name.replace("/", "_")
        sub_tag = f"_n{max_train_samples}" if max_train_samples else ""
        stem = f"{safe}_epoch{epoch}_{preprocessing.tag}{sub_tag}"
        out_file = output_dir / f"{stem}.json"

        if out_file.exists() and not force:
            try:
                with open(out_file) as f:
                    cached = json.load(f)
                # A cached score is only reusable if it came from the solver now being
                # asked for -- otherwise the run would silently mix solvers across
                # datasets. Results predating the solver field have no "solver" key and
                # were fit by a CPU sklearn probe that hit its iteration cap without
                # converging, so they re-fit here. That is cheap: the extracted features
                # are cached separately and get reused below.
                #
                # The metric set is gated the same way and for the same reason: a record
                # written before top-5 existed carries only top-1, and reusing it would
                # propagate a null top-5 that no later run ever repairs. Membership, not
                # truthiness -- a legitimate None (dataset with <= PROBE_TOP_K classes)
                # is a computed answer and counts as covered.
                #
                # The recipe markers close the same loophole for the protocol: records
                # from before the val-sweep existed were fit at a fixed C=1.0 and carry
                # no `c_selection`, and records from before standardization carry no
                # `feature_scaling`, so both re-fit here rather than mixing recipes.
                if (cached.get("solver") == probe_solver and "top5_accuracy" in cached
                        and is_current_recipe(cached)):
                    log.info("  [skip] %s (cached)", ds_name)
                    aggregated[ds_name] = cached
                    successes.append(ds_name)
                    continue
                log.info("  [refit] %s (cached solver=%s want %s; top-5 %s; C %s; scaling %s)",
                         ds_name, cached.get("solver"), probe_solver,
                         "present" if "top5_accuracy" in cached else "missing",
                         cached.get("c_selection") or "fixed",
                         cached.get("feature_scaling") or "raw")
            except Exception:
                log.warning("  [skip-stale] %s cache unreadable; re-running", ds_name)

        log.info("  [probe] %s", ds_name)
        try:
            train_feats_path = feat_dir / f"{stem}_train.npz"
            test_feats_path = feat_dir / f"{stem}_test.npz"

            if train_feats_path.exists() and test_feats_path.exists() and not force:
                tr = np.load(train_feats_path)
                te = np.load(test_feats_path)
                train_feats, train_labels = tr["feats"], tr["labels"]
                test_feats, test_labels = te["feats"], te["labels"]
                log.info("    using cached features")
            else:
                train_loader, test_loader = _build_dataset_loaders(
                    ds_name, transform, batch_size, num_workers,
                    data_root=data_root, download=download,
                    max_train_samples=max_train_samples,
                )
                train_feats, train_labels = _extract_features(model, train_loader, device, log, "train")
                test_feats, test_labels = _extract_features(model, test_loader, device, log, "test")
                _save_features(train_feats_path, train_feats, train_labels, log)
                _save_features(test_feats_path, test_feats, test_labels, log)

            n_features = train_feats.shape[1]
            const_mask = train_feats.std(axis=0) == 0
            n_const = int(const_mask.sum())
            if n_const > 0:
                keep = ~const_mask
                train_feats = train_feats[:, keep]
                test_feats = test_feats[:, keep]
                log.info("    dropped %d/%d constant dims; %d remain",
                         n_const, n_features, n_features - n_const)
            standardize = fit_standardizer(train_feats)
            train_feats = standardize(train_feats)
            test_feats = standardize(test_feats)

            best_c, sweep_info = _select_C(fit_and_score, train_feats, train_labels,
                                           log, device=device)
            acc, fit_info = fit_and_score(
                train_feats, train_labels, test_feats, test_labels, log,
                device=device, C=best_c, **fit_kwargs,
            )
            fit_info.pop("W", None)
            fit_info.pop("b", None)
            if not fit_info["converged"]:
                log.warning("    [not-converged] %s stopped at %d iters -- the score is a "
                            "lower bound on this checkpoint's probe accuracy",
                            ds_name, fit_info["n_iter"])

            block = {
                "epoch": str(epoch),
                "evaluation_type": "linear_probe",
                "dataset": ds_name,
                "score": acc,
                "metric": "top1_accuracy",
                "n_train": int(len(train_labels)),
                "max_train_samples": max_train_samples,
                "n_test": int(len(test_labels)),
                "n_features": n_features,
                "n_constant_dims_dropped": n_const,
                "preprocessing": preprocessing.tag,
                "c_selection": PROBE_C_SELECTION,
                "feature_scaling": PROBE_FEATURE_SCALING,
                "c_sweep": sweep_info,
                **fit_info,
            }
            with open(out_file, "w") as f:
                json.dump(block, f, indent=2)
            aggregated[ds_name] = block
            successes.append(ds_name)

        except Exception as exc:  # noqa: BLE001
            failures.append({"dataset": ds_name, "error": f"{type(exc).__name__}: {exc}"})
            err_path = out_file.with_suffix(".error.txt")
            with open(err_path, "w") as f:
                f.write(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
            log.exception("  [fail] %s", ds_name)

    summary = {
        "epoch": str(epoch),
        "datasets_requested": datasets,
        "solver": probe_solver,
        "preprocessing": preprocessing.tag,
        "successes": successes,
        "failures": failures,
    }
    with open(output_dir / f"_summary_epoch{epoch}_{preprocessing.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Linear probe: %d/%d succeeded.", len(successes), len(datasets))

    aggregated["_failures"] = failures
    return aggregated
