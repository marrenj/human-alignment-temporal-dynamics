import torch


class PerturbationStrategy:
    """Base strategy for applying a perturbation over a fixed epoch window."""
    def __init__(self, perturb_epoch: int, perturb_length: int, perturb_seed: int):
        """
        Args:
            perturb_epoch: First epoch (0-indexed) to apply the perturbation.
            perturb_length: Number of consecutive epochs to perturb.
            perturb_seed: Seed used to deterministically drive the perturbation.
        """
        self.start_epoch = perturb_epoch
        self.end_epoch = perturb_epoch + perturb_length - 1
        self.perturb_seed = perturb_seed

    def is_active_epoch(self, epoch_idx: int) -> bool:
        """Check if the perturbation should be applied in the given epoch (0-indexed)."""
        return self.start_epoch <= epoch_idx <= self.end_epoch

    def apply_to_batch(self, images, targets, device, epoch_idx: int, batch_idx: int):
        """Apply the perturbation to the batch (if active); return (images, targets)."""
        raise NotImplementedError


class TargetNoisePerturbation(PerturbationStrategy):
    """Injects Gaussian noise into targets during the active perturbation window."""
    def __init__(self, perturb_epoch, perturb_length, perturb_seed, target_mean=None, target_std=None):
        super().__init__(perturb_epoch, perturb_length, perturb_seed)
        self.target_mean = target_mean
        self.target_std = target_std

    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        if not self.is_active_epoch(epoch_idx):
            return images, targets  # no change outside perturbation window

        if self.target_mean is not None and self.target_std is not None:
            mean = self.target_mean.to(device=targets.device, dtype=targets.dtype)
            std = self.target_std.to(device=targets.device, dtype=targets.dtype)
        else:
            raise ValueError("target_mean and target_std must be provided for TargetNoisePerturbation")

        gen = torch.Generator(device=device)
        gen.manual_seed(self.perturb_seed + (epoch_idx + 1) * 1000 + batch_idx)

        noise = torch.randn(targets.shape, device=device, dtype=targets.dtype, generator=gen)
        random_targets = noise * std + mean
        return images, random_targets


class LabelShufflePerturbation(PerturbationStrategy):
    """Shuffles targets within a batch when the perturbation is active.

    Seed varies per epoch (``epoch_idx``) so the shuffle permutation refreshes
    each epoch in the window, giving stronger / higher-entropy noise injection
    than holding the permutation fixed across epochs.
    For CLIP the targets are text tokens, so the shuffle breaks image-text
    correspondence within the batch.
    """
    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        if not self.is_active_epoch(epoch_idx):
            return images, targets
        gen = torch.Generator(device=device)
        gen.manual_seed(self.perturb_seed + (epoch_idx + 1) * 1000 + batch_idx)
        shuffled = shuffle_targets(targets, generator=gen)
        return images, shuffled


class ImageNoisePerturbation(PerturbationStrategy):
    """Replaces images with Gaussian noise when active.

    Seed varies per epoch (``epoch_idx``) so the noise tensor differs each
    epoch in the window.
    """
    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        if not self.is_active_epoch(epoch_idx):
            return images, targets
        seed = self.perturb_seed + (epoch_idx + 1) * 1000 + batch_idx
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        # Create noise of the same shape as images (this will be drawn from N~(0,1), same as the normalized images)
        noise = torch.randn_like(images)
        return noise, targets


class UniformImagePerturbation(PerturbationStrategy):
    """Replaces images with constant 0.5 tensors when active."""
    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        if not self.is_active_epoch(epoch_idx):
            return images, targets
        uniform = torch.ones_like(images) * 0.5
        return uniform, targets


class NoPerturbation(PerturbationStrategy):
    """No-op perturbation used to disable perturbations cleanly."""
    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        return images, targets


def shuffle_targets(targets, generator=None):
    """
    Shuffle the target vectors along the batch dimension using the provided generator.
    """
    batch_size = targets.shape[0]
    perm = torch.randperm(batch_size, device=targets.device, generator=generator)
    return targets[perm]


def choose_perturbation_strategy(
    perturb_type,
    perturb_epoch,
    perturb_length,
    perturb_seed,
    target_mean=None,
    target_std=None,
):
    """
    Factory that returns the configured perturbation strategy.

    Args:
        perturb_type: One of random_target, label_shuffle, image_noise,
            uniform_images, none.
        perturb_epoch: First epoch (0-indexed) to apply the perturbation.
        perturb_length: Number of epochs to apply the perturbation.
        perturb_seed: Seed to drive deterministic perturbations.
        target_mean: Mean tensor for target noise (random_target).
        target_std: Std tensor for target noise (random_target).

    Returns:
        PerturbationStrategy: Configured strategy instance.
    """
    normalized_type = str(perturb_type).lower() if perturb_type is not None else 'none'

    if normalized_type == 'random_target':
        perturb_strategy = TargetNoisePerturbation(
            perturb_epoch=perturb_epoch,
            perturb_length=perturb_length,
            perturb_seed=perturb_seed,
            target_mean=target_mean,
            target_std=target_std
        )
    elif normalized_type == 'label_shuffle':
        perturb_strategy = LabelShufflePerturbation(perturb_epoch=perturb_epoch, perturb_length=perturb_length, perturb_seed=perturb_seed)
    elif normalized_type == 'image_noise':
        perturb_strategy = ImageNoisePerturbation(perturb_epoch=perturb_epoch, perturb_length=perturb_length, perturb_seed=perturb_seed)
    elif normalized_type == 'uniform_images':
        perturb_strategy = UniformImagePerturbation(perturb_epoch=perturb_epoch, perturb_length=perturb_length, perturb_seed=perturb_seed)
    elif normalized_type == 'none':
        perturb_strategy = NoPerturbation(perturb_epoch=perturb_epoch, perturb_length=perturb_length, perturb_seed=perturb_seed)
    else:
        raise ValueError(f"Perturbation type {perturb_type} not supported")
    return perturb_strategy