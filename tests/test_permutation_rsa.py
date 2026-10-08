"""Tests for condition-label permutation RSA p-values (Kriegeskorte et al., 2008)."""

import math
import numpy as np
import pytest
from scipy.stats import spearmanr, pearsonr, rankdata

from temporal_perturbations.inference.inference_core import (
    permutation_p_rdm,
    compute_rdm_similarity,
)


def _random_rdm_vec(n, rng):
    """Return a random upper-triangular vector for an n×n RDM."""
    m = n * (n - 1) // 2
    return rng.standard_normal(m)


class TestNConditionsDerivation:
    @pytest.mark.parametrize("n", [3, 10, 48, 100, 720])
    def test_valid_sizes(self, n):
        m = n * (n - 1) // 2
        vec = np.zeros(m)
        p = permutation_p_rdm(vec, vec, 0.0, "spearman", n_permutations=10, seed=0)
        assert isinstance(p, float)

    def test_invalid_size_raises(self):
        vec = np.zeros(5)  # not a valid n*(n-1)/2
        with pytest.raises(ValueError, match="not a valid upper-triangular size"):
            permutation_p_rdm(vec, vec, 0.0, "spearman", n_permutations=10)


class TestBijectionProperty:
    @pytest.mark.parametrize("n", [5, 20, 48])
    def test_permutation_is_bijection(self, n):
        """A condition-label permutation induces a bijection on upper-tri indices."""
        m = n * (n - 1) // 2
        pairs_i, pairs_j = np.triu_indices(n, k=1)
        rng = np.random.default_rng(42)
        for _ in range(50):
            perm = rng.permutation(n)
            pi = perm[pairs_i]
            pj = perm[pairs_j]
            a = np.minimum(pi, pj)
            b = np.maximum(pi, pj)
            new_idx = a * n - a * (a + 1) // 2 + b - a - 1
            assert len(set(new_idx)) == m, "Permutation did not produce a bijection"
            assert new_idx.min() >= 0
            assert new_idx.max() < m


class TestRankPermutationEquivalence:
    @pytest.mark.parametrize("n", [5, 20])
    def test_ranks_commute_with_bijective_reindex(self, n):
        """rankdata(v)[new_idx] == rankdata(v[new_idx]) when new_idx is a bijection."""
        m = n * (n - 1) // 2
        rng = np.random.default_rng(99)
        v = rng.standard_normal(m)
        pairs_i, pairs_j = np.triu_indices(n, k=1)
        for _ in range(50):
            perm = rng.permutation(n)
            pi, pj = perm[pairs_i], perm[pairs_j]
            a, b = np.minimum(pi, pj), np.maximum(pi, pj)
            new_idx = a * n - a * (a + 1) // 2 + b - a - 1
            np.testing.assert_array_equal(
                rankdata(v)[new_idx],
                rankdata(v[new_idx]),
            )


class TestPermutationPRdm:
    def test_identical_vectors_small_p(self):
        rng = np.random.default_rng(7)
        n = 20
        vec = _random_rdm_vec(n, rng)
        rho, _ = spearmanr(vec, vec)
        p = permutation_p_rdm(vec, vec, rho, "spearman", n_permutations=999, seed=42)
        assert p == pytest.approx(1 / 1000)  # (0 + 1) / (999 + 1)

    def test_uncorrelated_vectors_large_p(self):
        rng = np.random.default_rng(0)
        n = 30
        vec_a = _random_rdm_vec(n, rng)
        vec_b = _random_rdm_vec(n, rng)
        rho, _ = spearmanr(vec_a, vec_b)
        p = permutation_p_rdm(vec_a, vec_b, rho, "spearman",
                              n_permutations=999, seed=42)
        assert p > 0.01

    def test_seed_determinism(self):
        rng = np.random.default_rng(3)
        n = 15
        vec_a = _random_rdm_vec(n, rng)
        vec_b = _random_rdm_vec(n, rng)
        rho, _ = spearmanr(vec_a, vec_b)
        p1 = permutation_p_rdm(vec_a, vec_b, rho, "spearman",
                               n_permutations=500, seed=42)
        p2 = permutation_p_rdm(vec_a, vec_b, rho, "spearman",
                               n_permutations=500, seed=42)
        assert p1 == p2

    def test_phipson_smyth_floor(self):
        """P-value is always >= 1/(n_permutations+1), never zero."""
        rng = np.random.default_rng(1)
        n = 10
        vec = _random_rdm_vec(n, rng)
        rho, _ = spearmanr(vec, vec)
        n_perms = 200
        p = permutation_p_rdm(vec, vec, rho, "spearman",
                              n_permutations=n_perms, seed=42)
        assert p >= 1 / (n_perms + 1)
        assert p > 0

    def test_pearson_metric(self):
        rng = np.random.default_rng(5)
        n = 15
        vec_a = _random_rdm_vec(n, rng)
        vec_b = _random_rdm_vec(n, rng)
        rho, _ = pearsonr(vec_a, vec_b)
        p = permutation_p_rdm(vec_a, vec_b, rho, "pearson",
                              n_permutations=500, seed=42)
        assert 0 < p <= 1

    def test_invalid_metric_raises(self):
        vec = np.zeros(3)
        with pytest.raises(ValueError, match="Invalid similarity metric"):
            permutation_p_rdm(vec, vec, 0.0, "kendall")


class TestComputeRdmSimilarity:
    def test_returns_permutation_p(self):
        rng = np.random.default_rng(12)
        n = 15
        vec_a = _random_rdm_vec(n, rng)
        vec_b = _random_rdm_vec(n, rng)
        rho, p = compute_rdm_similarity(vec_a, vec_b, "spearman",
                                        n_permutations=500, permutation_seed=42)
        rho_ref, _ = spearmanr(vec_a, vec_b)
        assert rho == pytest.approx(rho_ref)
        assert 0 < p <= 1

    def test_pearson_path(self):
        rng = np.random.default_rng(8)
        n = 10
        vec_a = _random_rdm_vec(n, rng)
        vec_b = _random_rdm_vec(n, rng)
        rho, p = compute_rdm_similarity(vec_a, vec_b, "pearson",
                                        n_permutations=500, permutation_seed=42)
        rho_ref, _ = pearsonr(vec_a, vec_b)
        assert rho == pytest.approx(rho_ref)
        assert 0 < p <= 1
