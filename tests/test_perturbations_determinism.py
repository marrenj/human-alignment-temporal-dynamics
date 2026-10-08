import torch
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_perturbations.perturbations.perturbation_utils import (
    LabelShufflePerturbation,
    TargetNoisePerturbation,
)


def test_label_shuffle_matches_expected_seed_rule_and_is_deterministic() -> None:
    device = torch.device("cpu")
    batch = 8

    images = torch.zeros(batch, 3, 224, 224)
    targets = torch.arange(batch, dtype=torch.float32).unsqueeze(1)

    strat = LabelShufflePerturbation(perturb_epoch=5, perturb_length=2, perturb_seed=123)

    # Active epoch
    _, shuffled = strat.apply_to_batch(images, targets, device, epoch_idx=5, batch_idx=7)

    # Recompute expected permutation using the same seed formula used in the repo:
    # perturb_seed + (epoch_idx + 1) * 1000 + batch_idx (legacy formula — varies per epoch).
    gen = torch.Generator(device=device)
    gen.manual_seed(strat.perturb_seed + (5 + 1) * 1000 + 7)
    perm = torch.randperm(batch, generator=gen)
    expected = targets[perm]

    assert torch.equal(shuffled, expected)

    # Deterministic across repeated calls at the same epoch.
    _, shuffled2 = strat.apply_to_batch(images, targets, device, epoch_idx=5, batch_idx=7)
    assert torch.equal(shuffled2, shuffled)

    # Different epoch within window -> different permutation (legacy seed rule).
    _, shuffled_ep6 = strat.apply_to_batch(images, targets, device, epoch_idx=6, batch_idx=7)
    assert not torch.equal(shuffled_ep6, shuffled)

    # Inactive epoch -> unchanged
    _, unchanged = strat.apply_to_batch(images, targets, device, epoch_idx=4, batch_idx=7)
    assert torch.equal(unchanged, targets)


def test_target_noise_matches_expected_seed_rule_and_changes_with_batch_idx() -> None:
    device = torch.device("cpu")
    batch, dim = 4, 3

    images = torch.zeros(batch, 3, 224, 224)
    targets = torch.zeros(batch, dim, dtype=torch.float32)

    mean = torch.zeros(dim, dtype=torch.float32)
    std = torch.ones(dim, dtype=torch.float32)

    strat = TargetNoisePerturbation(
        perturb_epoch=0,
        perturb_length=1,
        perturb_seed=999,
        target_mean=mean,
        target_std=std,
    )

    _, noisy0 = strat.apply_to_batch(images, targets, device, epoch_idx=0, batch_idx=0)

    gen0 = torch.Generator(device=device)
    gen0.manual_seed(strat.perturb_seed + (0 + 1) * 1000 + 0)
    expected0 = torch.randn(targets.shape, generator=gen0)

    assert torch.allclose(noisy0, expected0)

    # Different batch_idx should change the sampled noise (in the repo's seed rule)
    _, noisy1 = strat.apply_to_batch(images, targets, device, epoch_idx=0, batch_idx=1)
    assert not torch.allclose(noisy1, noisy0)