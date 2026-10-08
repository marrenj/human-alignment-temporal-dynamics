"""Unit tests for the linear-probe core (temporal_perturbations.inference.linear_probe_eval).

Covers the top-k metric (against a brute-force reference, and at the class-count
boundary where it is undefined), cross-solver agreement between the GPU torch fit and
the sklearn CPU reference, and the per-dataset cache gate that decides when a stored
result may be reused.

All CPU, all synthetic -- no checkpoint, no download, no GPU.
"""

import json
import logging
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from temporal_perturbations.data.eval_transforms import IMAGENET_SUPERVISED_BILINEAR
from temporal_perturbations.inference import linear_probe_eval as lpe


LOG = logging.getLogger("test_linear_probe_eval")


def _separable(n_train, n_test, d, k, seed=0):
    """Well-conditioned k-class problem: labels are the argmax of a random linear map."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, k))
    Xtr = rng.normal(size=(n_train, d))
    Xte = rng.normal(size=(n_test, d))
    return Xtr, (Xtr @ W).argmax(1), Xte, (Xte @ W).argmax(1)


# _top_k_accuracy

def test_top_k_matches_bruteforce_sort():
    """argpartition must agree exactly with a full sort of the score matrix."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(500, 10))
    labels = rng.integers(0, 10, 500)
    classes = np.arange(10)

    reference = float(
        (np.argsort(-scores, axis=1)[:, :lpe.PROBE_TOP_K] == labels[:, None]).any(1).mean()
    )
    assert lpe._top_k_accuracy(scores, classes, labels) == pytest.approx(reference)


def test_top_k_respects_nonconsecutive_class_labels():
    """`classes` is sklearn/np.unique order, not necessarily 0..k-1."""
    rng = np.random.default_rng(1)
    classes = np.array([3, 17, 42, 55, 61, 99])
    scores = rng.normal(size=(200, len(classes)))
    labels = classes[rng.integers(0, len(classes), 200)]

    reference = float(
        (classes[np.argsort(-scores, axis=1)[:, :lpe.PROBE_TOP_K]] == labels[:, None])
        .any(1).mean()
    )
    assert lpe._top_k_accuracy(scores, classes, labels) == pytest.approx(reference)


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_top_k_undefined_at_or_below_k_classes(n_classes):
    """None, not a trivially-perfect 1.0, so an aggregate table shows N/A."""
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(50, n_classes))
    labels = rng.integers(0, n_classes, 50)
    assert lpe._top_k_accuracy(scores, np.arange(n_classes), labels) is None


def test_top_k_defined_just_above_k_classes():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(50, lpe.PROBE_TOP_K + 1))
    labels = rng.integers(0, lpe.PROBE_TOP_K + 1, 50)
    assert lpe._top_k_accuracy(scores, np.arange(lpe.PROBE_TOP_K + 1), labels) is not None


def test_top_k_at_chance_is_k_over_n_classes():
    """Random scores over 10 classes: top-5 ~= 0.5, top-1 ~= 0.1.

    Catches an off-by-one in the argpartition slice, which would shift top-5 to ~0.4
    or ~0.6 while leaving every other assertion here green.
    """
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(20000, 10))
    labels = rng.integers(0, 10, 20000)
    assert lpe._top_k_accuracy(scores, np.arange(10), labels) == pytest.approx(0.5, abs=0.02)


# Solver agreement

@pytest.mark.parametrize("C", [1.0, 1e-2])
def test_solvers_agree_on_both_metrics(C):
    """torch and sklearn minimize the same objective, so they must land on the same fit.

    Parametrized over C because the sweep now fits at non-default values -- both
    solvers must apply the SAME C or the sweep selects on one objective and the
    final fit runs another. The equivalence criterion is the OBJECTIVE: a wrong C
    on either side moves it by orders of magnitude more than the 1e-6 bound.
    Accuracy gets a one-test-sample allowance -- at strongly regularized C the
    logits are compressed, so a borderline sample can sit inside the gradient-
    tolerance ball and land on either side of the argmax.
    """
    Xtr, ytr, Xte, yte = _separable(600, 300, d=12, k=10, seed=5)

    acc_sk, info_sk = lpe._fit_and_score(Xtr, ytr, Xte, yte, LOG, max_iter=5000, C=C)
    acc_th, info_th = lpe._fit_and_score_torch(Xtr, ytr, Xte, yte, LOG,
                                               device="cpu", max_iter=5000, C=C)

    one_sample = 1.0 / len(yte)
    assert acc_sk == pytest.approx(acc_th, abs=one_sample + 1e-9)
    assert info_sk["top5_accuracy"] == pytest.approx(info_th["top5_accuracy"],
                                                     abs=one_sample + 1e-9)
    assert info_sk["objective"] == pytest.approx(info_th["objective"], rel=1e-6)
    assert info_sk["n_classes"] == info_th["n_classes"] == 10
    assert info_sk["C"] == info_th["C"] == C


@pytest.mark.parametrize("fit", [lpe._fit_and_score, lpe._fit_and_score_torch])
def test_top5_at_least_top1(fit):
    Xtr, ytr, Xte, yte = _separable(400, 200, d=10, k=8, seed=6)
    acc, info = fit(Xtr, ytr, Xte, yte, LOG, device="cpu", max_iter=3000)
    assert info["top5_accuracy"] >= acc


def test_torch_solver_handles_binary():
    """The k-class softmax path must not choke when k == 2; top-5 is undefined there."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] > 0).astype(int)

    acc, info = lpe._fit_and_score_torch(X, y, X, y, LOG, device="cpu", max_iter=2000)
    assert info["n_classes"] == 2
    assert info["top5_accuracy"] is None
    assert acc > 0.9


@pytest.mark.xfail(raises=IndexError, strict=True,
                   reason="pre-existing: _penalized_nll assumes a k-column coef_, but "
                          "sklearn fits a single logit vector for k=2. Affects only the "
                          "two binary datasets in PAPER_PROBE_DATASETS under "
                          "--probe_solver sklearn; the default torch solver is unaffected.")
def test_sklearn_solver_binary_objective():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] > 0).astype(int)
    lpe._fit_and_score(X, y, X, y, LOG, max_iter=2000)


# _extract_features

def test_extract_features_copies_labels_out_of_loader_tensors(monkeypatch):
    """Every per-batch label array must OWN its data, not view a loader tensor.

    `np.asarray(lbls)` returned a view over the tensor's storage -- the tensor object
    dies, the storage does not -- so with `pin_memory=True` the labels list kept every
    batch's pinned buffer alive and PyTorch's caching host allocator never recycled
    them: the pinned pool grew one batch (~150 MB) per iteration until the kernel
    OOM-killed the process on PCam (167 GB). The leak only manifests at real batch
    sizes and forward-pass timing, so this pins the invariant rather than the symptom;
    the per-batch arrays are intercepted at the final concatenate, which is the only
    point they are observable.
    """
    captured = []
    real_concatenate = np.concatenate

    def spy(arrays, axis=0):
        captured.append(list(arrays))
        return real_concatenate(arrays, axis=axis)

    monkeypatch.setattr(np, "concatenate", spy)

    class Loader:
        def __len__(self):
            return 3

        def __iter__(self):
            for _ in range(3):
                yield [torch.randn(4, 3), torch.arange(4)]

    feats, labels = lpe._extract_features(nn.Identity(), Loader(), "cpu", LOG)
    assert feats.shape == (12, 3) and labels.shape == (12,)
    label_arrays = captured[1]
    assert not torch.arange(4).numpy().flags.owndata   # the view this guards against
    assert all(a.flags.owndata for a in label_arrays), "label arrays view loader tensors"


# C selection (_val_split / _select_C)

def test_val_split_is_stratified():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(1000, 4))
    y = np.repeat(np.arange(10), 100)

    x_fit, x_val, y_fit, y_val = lpe._val_split(X, y)
    assert len(y_val) == int(lpe.PROBE_VAL_FRACTION * 1000)
    assert len(y_fit) + len(y_val) == 1000
    counts = np.bincount(y_val, minlength=10)
    assert (counts == counts[0]).all()


def test_val_split_singleton_goes_to_fit():
    """Singleton classes land in fit, not val; the rest stays stratified."""
    rng = np.random.default_rng(10)
    X = rng.normal(size=(101, 4))
    y = np.array([0] * 50 + [1] * 50 + [2])

    x_fit, x_val, y_fit, y_val = lpe._val_split(X, y)
    assert len(y_fit) + len(y_val) == 101
    assert 2 not in y_val
    assert 2 in y_fit


def _stub_fit(val_acc_of_c):
    """A fit function whose val accuracy is a pure function of C -- no real fitting."""
    def fit(train_feats, train_labels, test_feats, test_labels, logger, *,
            device=None, max_iter=None, C=None, **_):
        assert max_iter == lpe.PROBE_SWEEP_MAX_ITER  # sweep fits are always capped
        return val_acc_of_c(C), {"n_iter": 1, "converged": True, "C": C}
    return fit


def _fake_train(n=100, d=3, k=2, seed=11):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.integers(0, k, n)


def test_select_C_finds_and_refines_the_peak():
    """Peaked val-accuracy profile: the winner must be within one refined step
    (half a coarse decade) of the true optimum, and refinement must have evaluated
    midpoints rather than just the coarse grid."""
    X, y = _fake_train()
    peak_log_c = -2.0
    best_c, sweep = lpe._select_C(
        _stub_fit(lambda c: 1.0 - abs(math.log10(c) - peak_log_c) / 10), X, y, LOG,
        device="cpu")

    assert abs(math.log10(best_c) - peak_log_c) <= 0.5
    assert len(sweep["C_grid"]) > len(lpe.PROBE_C_GRID)
    assert sweep["C_grid"] == sorted(sweep["C_grid"])
    assert len(sweep["val_top1"]) == len(sweep["C_grid"])


def test_select_C_tie_breaks_toward_stronger_regularization():
    """Flat profile (degenerate features often produce one): smallest C must win."""
    X, y = _fake_train()
    best_c, _ = lpe._select_C(_stub_fit(lambda c: 0.5), X, y, LOG, device="cpu")
    assert best_c == pytest.approx(min(lpe.PROBE_C_GRID))


# Cache gate

def _write_cached_block(probe_dir, dataset, epoch, *, solver, with_top5,
                        c_selection=lpe.PROBE_C_SELECTION,
                        feature_scaling=lpe.PROBE_FEATURE_SCALING):
    stem = f"{dataset}_epoch{epoch}_{IMAGENET_SUPERVISED_BILINEAR.tag}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    block = {"epoch": str(epoch), "evaluation_type": "linear_probe", "dataset": dataset,
             "score": 0.5, "metric": "top1_accuracy", "solver": solver}
    if with_top5:
        block["top5_accuracy"] = 0.9
    if c_selection is not None:
        block["c_selection"] = c_selection
    if feature_scaling is not None:
        block["feature_scaling"] = feature_scaling
    (probe_dir / f"{stem}.json").write_text(json.dumps(block))


def _run_probe_on_cache(tmp_path, **kwargs):
    """Drive run_linear_probe with no usable data root, so only the cache gate can succeed.

    A gate hit returns before any loader is built; a gate miss falls through to
    _build_dataset_loaders, which raises for the torchvision datasets when data_root is
    None and is recorded as a per-dataset failure.
    """
    return lpe.run_linear_probe(
        nn.Identity(), output_dir=tmp_path, epoch=7,
        preprocessing=IMAGENET_SUPERVISED_BILINEAR, datasets=["cifar10"],
        device="cpu", data_root=None, download=False, logger=LOG, **kwargs,
    )


def test_cache_hit_when_solver_and_top5_present(tmp_path):
    _write_cached_block(tmp_path, "cifar10", 7, solver="torch", with_top5=True)
    results = _run_probe_on_cache(tmp_path)
    assert results["cifar10"]["top5_accuracy"] == 0.9

    summary = json.loads(
        (tmp_path / f"_summary_epoch7_{IMAGENET_SUPERVISED_BILINEAR.tag}.json").read_text())
    assert summary["successes"] == ["cifar10"] and not summary["failures"]


def test_cache_miss_when_top5_absent(tmp_path):
    """A pre-top-5 record must not be reused, or its null top-5 never self-heals."""
    _write_cached_block(tmp_path, "cifar10", 7, solver="torch", with_top5=False)
    results = _run_probe_on_cache(tmp_path)
    assert "cifar10" not in results

    summary = json.loads(
        (tmp_path / f"_summary_epoch7_{IMAGENET_SUPERVISED_BILINEAR.tag}.json").read_text())
    assert summary["successes"] == []
    assert summary["failures"][0]["dataset"] == "cifar10"


def test_cache_miss_on_solver_change(tmp_path):
    _write_cached_block(tmp_path, "cifar10", 7, solver="sklearn", with_top5=True)
    assert "cifar10" not in _run_probe_on_cache(tmp_path, probe_solver="torch")


def test_cache_miss_on_pre_sweep_record(tmp_path):
    """A record from before the C-sweep existed (fixed C=1.0, no marker) must re-fit."""
    _write_cached_block(tmp_path, "cifar10", 7, solver="torch", with_top5=True,
                        c_selection=None)
    assert "cifar10" not in _run_probe_on_cache(tmp_path)


def test_cache_miss_on_pre_standardization_record(tmp_path):
    """A record fit on raw features (no feature_scaling tag) must re-fit."""
    _write_cached_block(tmp_path, "cifar10", 7, solver="torch", with_top5=True,
                        feature_scaling=None)
    assert "cifar10" not in _run_probe_on_cache(tmp_path)


# Feature standardization

def test_standardizer_uses_train_statistics():
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 3.0, size=(500, 4)).astype(np.float32)
    test = rng.normal(5.0, 3.0, size=(200, 4)).astype(np.float32)
    standardize = lpe.fit_standardizer(train)
    z_train, z_test = standardize(train), standardize(test)
    assert z_train.dtype == np.float32
    np.testing.assert_allclose(z_train.mean(0), 0.0, atol=1e-5)
    np.testing.assert_allclose(z_train.std(0), 1.0, atol=1e-4)
    mu, sd = train.mean(0, dtype=np.float64), train.std(0, dtype=np.float64)
    np.testing.assert_allclose(z_test, (test - mu) / sd, rtol=1e-4, atol=1e-5)


def test_standardizer_floor_keeps_noise_units_small():
    """A unit whose std is float32 noise must not be amplified to unit scale."""
    rng = np.random.default_rng(0)
    train = np.column_stack([rng.normal(0, 1, 1000), 1.0 + rng.normal(0, 1e-7, 1000)])
    z = lpe.fit_standardizer(train)(train)
    assert z[:, 1].std() < 1e-3


def test_standardizer_accepts_integer_pixels():
    pixels = np.arange(24, dtype=np.uint8).reshape(6, 4)
    z = lpe.fit_standardizer(pixels)(pixels)
    assert np.issubdtype(z.dtype, np.floating)
    np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-6)


def test_standardizing_unsticks_a_small_gradient_fit():
    """Class signal present, but in raw units so small that max |grad| at zero weights is
    below PROBE_TOL. The raw fit stops at iteration 0 and scores chance; the standardized
    fit learns the classes."""
    rng = np.random.default_rng(0)
    k, d, n_per = 50, 20, 40
    centers = rng.normal(0, 1, size=(k, d))
    y = np.repeat(np.arange(k), n_per)
    X = (centers[y] + rng.normal(0, 0.5, size=(len(y), d))) * 1e-4
    y_te = np.repeat(np.arange(k), 10)
    X_te = (centers[y_te] + rng.normal(0, 0.5, size=(len(y_te), d))) * 1e-4

    raw_acc, raw_info = lpe._fit_and_score_torch(X, y, X_te, y_te, LOG, device="cpu", C=1.0)
    assert raw_info["n_iter"] == 0 and raw_acc < 3.0 / k

    standardize = lpe.fit_standardizer(X)
    acc, info = lpe._fit_and_score_torch(standardize(X), y, standardize(X_te), y_te, LOG,
                                         device="cpu", C=1.0)
    assert info["converged"] and acc > 0.9
