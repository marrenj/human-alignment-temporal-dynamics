"""Unit tests for the layer-reweighted RSA engine (temporal_perturbations.inference.frrsa_core).

Covers the model-free fitting math (planted-signal recovery, CV non-leakage,
the in-sample>CV bias that motivates nested CV), the feature-pooling rule, the
architecture layer selector, and an end-to-end extract->RDM->fit smoke test.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from temporal_perturbations.inference import frrsa_core as fc


# Helpers

def _random_rdm_vectors(n_conditions, n_layers, seed=0):
    """n_layers random upper-tri 'RDM' vectors of the right length for n_conditions."""
    rng = np.random.RandomState(seed)
    n_pairs = n_conditions * (n_conditions - 1) // 2
    return {f"layer{j}": rng.rand(n_pairs) for j in range(n_layers)}


# Pooling rule

def test_pool_conv_4d_flattens_spatial():
    out = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    pooled = fc._pool_layer_output(out)
    assert pooled.shape == (2, 3 * 4 * 4)          # flatten C*H*W, keep spatial layout
    assert torch.equal(pooled, out.reshape(2, -1))


def test_pool_transformer_3d_takes_cls_token():
    out = torch.randn(2, 5, 7)                      # [N, tokens, D]
    pooled = fc._pool_layer_output(out)
    assert pooled.shape == (2, 7)                   # CLS = token index 0
    assert torch.equal(pooled, out[:, 0, :])


def test_pool_2d_passthrough_and_tuple_output():
    out2d = torch.randn(2, 9)
    assert torch.equal(fc._pool_layer_output(out2d), out2d)
    tup = (torch.randn(2, 3, 4, 4), torch.randn(2, 5))
    assert fc._pool_layer_output(tup).shape == (2, 48)   # first element, flattened


# Leave-conditions-out CV splits

def test_splits_partition_conditions():
    n, k = 50, 5
    splits = fc.leave_conditions_out_splits(n, k, seed=1)
    assert len(splits) == k
    covered = np.concatenate([test for _, test in splits])
    assert sorted(covered.tolist()) == list(range(n))     # every condition tested once
    for train, test in splits:
        assert set(train).isdisjoint(set(test))           # disjoint
        assert len(train) + len(test) == n


def test_no_test_condition_leaks_into_training_pairs():
    """A both-in-train pair must never touch a held-out (test) condition."""
    n = 40
    rows, cols = fc._triu_endpoints(n)
    for train, test in fc.leave_conditions_out_splits(n, 5, seed=2):
        train_pairs = fc._both_in_mask(train, rows, cols, n)
        test_conds = set(test.tolist())
        touched = set(rows[train_pairs]) | set(cols[train_pairs])
        assert touched.isdisjoint(test_conds)


# nnls_ridge

def test_nnls_ridge_is_nonnegative_and_shrinks_with_alpha():
    rng = np.random.RandomState(0)
    X = rng.rand(200, 4)
    y = X @ np.array([1.0, 0.0, 2.0, 0.0]) + 0.01 * rng.randn(200)
    w0 = fc.nnls_ridge(X, y, alpha=0.0)
    w_big = fc.nnls_ridge(X, y, alpha=1e4)
    assert np.all(w0 >= 0) and np.all(w_big >= 0)
    assert np.linalg.norm(w_big) < np.linalg.norm(w0)     # heavy ridge shrinks weights


# fit_frrsa — planted signal recovery

def test_fit_frrsa_recovers_planted_layer():
    n = 50
    layers = _random_rdm_vectors(n, n_layers=4, seed=3)
    target = 3.0 * layers["layer2"]                        # target IS layer2's RDM (scaled)
    res = fc.fit_frrsa(layers, target, n_conditions=n, scoring="both", seed=7)

    # Weight concentrates on the planted layer, and both scores are near-perfect.
    top = max(res["layer_weights_normalized"], key=res["layer_weights_normalized"].get)
    assert top == "layer2"
    assert res["layer_weights_normalized"]["layer2"] > 0.9
    assert res["cv_score"] > 0.9
    assert res["insample_score"] > 0.9


def test_fit_frrsa_insample_inflates_above_cv_on_noise():
    """The rationale for nested CV: with many layers and a noise target, the
    in-sample score overfits above the honest held-out CV score."""
    n = 40
    layers = _random_rdm_vectors(n, n_layers=15, seed=11)
    rng = np.random.RandomState(99)
    n_pairs = n * (n - 1) // 2
    target = rng.rand(n_pairs)                             # unrelated to any layer
    res = fc.fit_frrsa(layers, target, n_conditions=n, scoring="both", seed=5)
    assert res["cv_score"] < 0.2                           # no real signal to generalize
    assert res["insample_score"] > res["cv_score"]         # but in-sample overfits upward


def test_fit_frrsa_output_schema():
    n = 30
    layers = _random_rdm_vectors(n, n_layers=3, seed=1)
    res = fc.fit_frrsa(layers, layers["layer0"], n_conditions=n)
    for key in ("cv_score", "cv_score_std", "insample_score", "alpha",
                "layer_weights", "layer_weights_normalized", "n_layers", "dropped_layers"):
        assert key in res
    # Default scoring is the standard 'insample' -> the nested CV is skipped.
    assert res["cv_score"] is None and res["n_cv_folds"] == 0
    assert np.isfinite(res["insample_score"])
    assert pytest.approx(1.0, abs=1e-6) == sum(res["layer_weights_normalized"].values()) or \
        sum(res["layer_weights_normalized"].values()) == 0.0


def test_fit_frrsa_records_dropped_duplicate_layers():
    """Near-identical layers (corr > 0.999) are collapsed and reported in dropped_layers."""
    n = 40
    layers = _random_rdm_vectors(n, n_layers=3, seed=4)
    layers["layer2"] = layers["layer0"].copy()          # exact duplicate of layer0
    res = fc.fit_frrsa(layers, layers["layer1"], n_conditions=n, seed=0)
    assert res["dropped_layers"] == ["layer2"]          # later duplicate dropped
    assert "layer2" not in res["layer_weights"]         # and absent from the surviving weights
    assert res["n_layers"] == 2


# compute_layer_rdm_vector

def test_compute_layer_rdm_vector_shape_and_range():
    rng = np.random.RandomState(0)
    feats = rng.rand(12, 40)
    vec = fc.compute_layer_rdm_vector(feats, metric="cosine")
    assert vec.shape == (12 * 11 // 2,)
    assert np.all(vec >= -1e-6) and np.all(vec <= 2 + 1e-6)   # cosine distance in [0, 2]


def test_compute_layer_rdm_vector_constant_rows_no_nan():
    feats = np.ones((8, 5))                                # all identical -> 0 distance, no NaN
    vec = fc.compute_layer_rdm_vector(feats, metric="cosine")
    assert not np.any(np.isnan(vec))


# select_layers — architecture dispatch (structural stubs)

def test_select_layers_clip_visual_stub():
    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
            self.ln_post = nn.Identity()

    class ClipStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()

    labels = [lab for lab, _ in fc.select_layers(ClipStub(), architecture="ViT-B/32")]
    assert labels == ["visual.transformer.0", "visual.transformer.1",
                      "visual.transformer.2", "visual.ln_post"]


def test_select_layers_vit_stub():
    class ViTStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.norm = nn.LayerNorm(4)

    labels = [lab for lab, _ in fc.select_layers(ViTStub(), architecture="ViT-B/16")]
    assert labels == ["blocks.0", "blocks.1", "norm"]


def test_select_layers_resnet_stub_is_block_level():
    def stage(n):
        return nn.Sequential(*[nn.Identity() for _ in range(n)])

    class ResNetStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.act1 = nn.Identity()
            self.layer1, self.layer2, self.layer3, self.layer4 = stage(2), stage(2), stage(3), stage(2)
            self.global_pool = nn.Identity()

    labels = [lab for lab, _ in fc.select_layers(ResNetStub(), architecture="RN50")]
    assert labels == ["stem",
                      "layer1.0", "layer1.1",
                      "layer2.0", "layer2.1",
                      "layer3.0", "layer3.1", "layer3.2",
                      "layer4.0", "layer4.1",
                      "global_pool"]


def test_select_layers_unrecognized_raises():
    with pytest.raises(ValueError, match="unrecognised model structure"):
        fc.select_layers(nn.Linear(3, 3), architecture="mystery")


# End-to-end: FeatureExtractor -> per-layer RDM -> fit

def test_end_to_end_extract_rdm_fit():
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
            self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 6)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    torch.manual_seed(0)
    model = TinyCNN().eval()
    n_cond = 20
    images = torch.randn(n_cond, 3, 8, 8)

    layer_defs = [("conv1", model.conv1), ("conv2", model.conv2), ("fc", model.fc)]
    with fc.FeatureExtractor(layer_defs) as extractor, torch.no_grad():
        model(images)
        feats = extractor.pop()

    assert feats["conv1"].shape == (n_cond, 4 * 8 * 8)     # conv flattened
    assert feats["fc"].shape == (n_cond, 6)

    layer_rdms = {name: fc.compute_layer_rdm_vector(f.numpy(), metric="cosine")
                  for name, f in feats.items()}
    res = fc.fit_frrsa(layer_rdms, layer_rdms["conv2"], n_conditions=n_cond, seed=0)
    assert np.isfinite(res["insample_score"])
    assert res["n_layers"] >= 1
