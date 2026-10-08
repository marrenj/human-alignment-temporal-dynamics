"""Canonical eval-time image preprocessing, per training recipe and architecture.

Which normalization and crop geometry a checkpoint expects is a property of the
**pipeline that trained it**, not of its architecture family. Getting it wrong is
silent: the model still runs, the features still look reasonable, and only the
numbers are off. This module is the single place those choices are recorded, so
every eval agrees and the provenance stays checkable.

Provenance -- the `runs_root` checkpoints were trained by
`pipeline/scripts/training/train_from_scratch.py`, which branches on `task`:
"supervised" builds `ImageNetDataModule` (`scratch_training/dataset_imagenet.py`),
anything else builds `YFCC15MDataModule` (`scratch_training/dataset.py`). Each run's `hparams.yaml` records which branch it took via the
`training_config` type, and the mean/std are module-level constants in those two
files with no per-architecture override:

| architecture   | training_config                  | normalization | eval geometry             |
| -------------- | -------------------------------- | ------------- | ------------------------- |
| RN50           | SupervisedImageNetTrainingConfig  | ImageNet      | Resize(256) -> Crop(224)  |
| ViT-B/16       | SupervisedImageNetTrainingConfig  | ImageNet      | Resize(256) -> Crop(224)  |
| ConvNeXt-B     | SupervisedImageNetTrainingConfig  | ImageNet      | Resize(256) -> Crop(224)  |
| CLIP ViT-B/32  | DeCLIPLitDataTrainingConfig       | CLIP          | Resize(224) -> Crop(224)  |

Interpolation is per-run, read from each `hparams.yaml`: RN50 trained with
`bilinear`, the other three with `bicubic`. (RandAugment is recorded there too --
`rand-m9-mstd0.5-inc1` for ViT-B/16 and ConvNeXt-B -- but it is train-time only and
so plays no part in an eval transform.)

Two consequences worth stating plainly:

  * Applying the CLIP transform to the three ImageNet-supervised models is wrong
    twice over. CLIP's std is 17-23% larger than ImageNet's, so inputs arrive at
    ~0.82-0.86x the amplitude the model was trained on; and `Resize(224)` ahead of
    `CenterCrop(224)` makes the crop a no-op, where training used `Resize(256)` --
    a 12.5% center zoom, i.e. a different object scale.
  * The reverse is equally wrong. The CLIP checkpoint really does want
    `Resize(224)` and the CLIP statistics; do not "fix" it onto the ImageNet recipe.

`tag` identifies the recipe in cache paths. Extracted features depend on the
preprocessing, but caches are otherwise keyed only on dataset + epoch, so without
it a corrected run would silently reuse features from the old one.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_INTERPOLATION = {
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "bilinear": transforms.InterpolationMode.BILINEAR,
}


def to_rgb(img):
    """Convert a PIL image to RGB, passing tensors through untouched.

    Module-level rather than an inline lambda because DataLoader workers on Windows
    and macOS start via *spawn*, which pickles the dataset -- and its transform along
    with it. A lambda is unpicklable and fails the whole run there, while being
    perfectly fine under Linux's fork.
    """
    return img.convert("RGB") if hasattr(img, "convert") else img


@dataclass(frozen=True)
class EvalPreprocessing:
    """One eval-time transform recipe: normalization + resize geometry.

    `resize_mode` is either:
      * "shorter_side" -- resize the shorter side to `image_size * resize_ratio`,
        preserving aspect ratio, then center-crop to `image_size`. What both training
        pipelines do (ratio 256/224 supervised, 1.0 for CLIP).
      * "squash"       -- resize straight to (image_size, image_size), distorting
        aspect ratio and keeping the whole frame. `resize_ratio` is unused. This is
        what the THINGS/NOD dataset classes have always done; see THINGS_LEGACY.
    """

    tag: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize_ratio: float
    interpolation: str
    resize_mode: str = "shorter_side"

    def build(self, image_size: int = 224) -> transforms.Compose:
        interp = _INTERPOLATION[self.interpolation]
        if self.resize_mode == "squash":
            geometry = [transforms.Resize((image_size, image_size), interpolation=interp)]
        elif self.resize_mode == "shorter_side":
            geometry = [
                transforms.Resize(round(image_size * self.resize_ratio), interpolation=interp),
                transforms.CenterCrop(image_size),
            ]
        else:
            raise ValueError(f"Unknown resize_mode {self.resize_mode!r}")

        # RGB conversion comes first to mirror the training pipelines, which convert
        # at decode time (`_decode_image` in dataset_imagenet.py) -- so a palette or
        # grayscale source is resampled in RGB there and here alike.
        return transforms.Compose(
            [transforms.Lambda(to_rgb)] + geometry
            + [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )


IMAGENET_SUPERVISED_BILINEAR = EvalPreprocessing(
    "in1k_bilinear", IMAGENET_MEAN, IMAGENET_STD, 256 / 224, "bilinear")
IMAGENET_SUPERVISED_BICUBIC = EvalPreprocessing(
    "in1k_bicubic", IMAGENET_MEAN, IMAGENET_STD, 256 / 224, "bicubic")
CLIP_CONTRASTIVE = EvalPreprocessing(
    "clip", CLIP_MEAN, CLIP_STD, 1.0, "bicubic")

# THINGS image-set statistics with an aspect-squashing resize -- what
# ThingsBehavioralDataset / ThingsFMRIDataset / NodDataset have always applied, and
# therefore what every CLIP-HBA number in this repo was produced under. It matches no
# training pipeline; it is preserved verbatim so CLIP-HBA results stay reproducible and
# so the cross-architecture sweeps can reproduce their own pre-fix numbers on request.
# Bilinear because that is torchvision's `Resize` default, which those classes rely on.
THINGS_LEGACY = EvalPreprocessing(
    "things_legacy", (0.52997664, 0.48070561, 0.41943838),
    (0.27608301, 0.26593025, 0.28238822), 1.0, "bilinear", resize_mode="squash")

# Keys match `run_model_to_model_rsa.ARCH_PREFIX` (note the CLIP entry
# has a space, not a hyphen), which is what the eval drivers pass through.
ARCH_EVAL_PREPROCESSING = {
    "RN50": IMAGENET_SUPERVISED_BILINEAR,
    "ViT-B/16": IMAGENET_SUPERVISED_BICUBIC,
    "ConvNeXt-B": IMAGENET_SUPERVISED_BICUBIC,
    "CLIP ViT-B/32": CLIP_CONTRASTIVE,
}


PREPROCESSING_BY_TAG = {
    p.tag: p for p in (IMAGENET_SUPERVISED_BILINEAR, IMAGENET_SUPERVISED_BICUBIC,
                       CLIP_CONTRASTIVE, THINGS_LEGACY)
}

# Drivers expose this as their --preprocessing choices: "auto" resolves per
# architecture, the rest name a recipe outright (mainly to reproduce old numbers).
PREPROCESSING_CHOICES = ["auto", *sorted(PREPROCESSING_BY_TAG)]


def eval_preprocessing(architecture: str) -> EvalPreprocessing:
    """The recipe an `runs_root` architecture's checkpoints were trained under."""
    try:
        return ARCH_EVAL_PREPROCESSING[architecture]
    except KeyError:
        raise KeyError(
            f"No eval preprocessing recorded for architecture {architecture!r}. "
            f"Known: {sorted(ARCH_EVAL_PREPROCESSING)}. Add it here rather than "
            f"guessing at the call site -- the correct answer comes from the run's "
            f"hparams.yaml `training_config` type (see this module's docstring)."
        ) from None


def frrsa_result_filename(dataset, epoch, preprocessing,
                          bn_recalibrated=False) -> str:
    """Result filename for one FR-RSA (dataset, epoch, preprocessing).

    Shared deliberately, unlike the other drivers' per-consumer tagging rules: here
    `inference_core` WRITES the file and `run_frrsa.py` tests for it (to skip completed
    work, and to confirm output was produced). If those drift, the driver skips
    everything as "already present".

    The tag sits BEFORE `_epoch` because `frrsa_fig_data.py` globs
    `..._frrsa_epoch*.json`, which a trailing tag would match; legacy stays untagged so
    every existing reader keeps resolving its exact filename.
    """
    tag = "" if preprocessing.tag == THINGS_LEGACY.tag else f"_{preprocessing.tag}"
    bn = "_bnrecal" if bn_recalibrated else ""
    return f"inference_results_{dataset}_frrsa{tag}{bn}_epoch{epoch}.json"


def resolve_preprocessing(name, architecture=None, training_mode=None) -> EvalPreprocessing:
    """Turn a driver's --preprocessing value into a recipe.

    "auto" (or None) means "whatever this checkpoint was trained under". Any other
    value names a recipe in PREPROCESSING_BY_TAG -- chiefly `things_legacy`, to
    regenerate results matching the pre-fix runs.

    `training_mode` is consulted FIRST when resolving "auto", because architecture
    names alone are ambiguous about the training pipeline. The FR-RSA drivers label the
    contrastive CLIP family `"ViT-B/32"` (while the m2m sweep calls it
    `"CLIP ViT-B/32"`), and a bare "ViT-B/32" says nothing about whether it was trained
    contrastively or supervised -- `training_mode` does.
    """
    if name not in (None, "auto"):
        try:
            return PREPROCESSING_BY_TAG[name]
        except KeyError:
            raise KeyError(
                f"Unknown preprocessing {name!r}. Known: {sorted(PREPROCESSING_BY_TAG)}, "
                f"or 'auto' to resolve from the architecture / training_mode."
            ) from None

    if training_mode == "clip_contrastive":
        return CLIP_CONTRASTIVE
    if training_mode == "scratch":
        # Trained in THIS repo by data/imagenet_dataset.py: ImageNet stats,
        # Resize(256) -> CenterCrop(224), torchvision's default bilinear.
        return IMAGENET_SUPERVISED_BILINEAR
    # `scratch_lightning` (the runs_root timm checkpoints) and anything unlabelled
    # resolve per architecture, where the interpolation actually differs per run.
    if architecture is None:
        raise ValueError(
            "preprocessing='auto' needs an architecture or a training_mode to resolve "
            "against; got neither."
        )
    return eval_preprocessing(architecture)
