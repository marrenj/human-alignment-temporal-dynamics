"""timm-backed supervised image classifiers for the ImageNet-1K perturbation runs.

The CLIP path (model.py) is image-text contrastive; these are plain N-way
classifiers (ViT-B/16, ConvNeXt-B, ResNet50) used for the supervised arm of the
matrix. Kept deliberately thin: timm builds the backbone + head, this module only
standardizes construction from a SupervisedModelConfig.
"""
from __future__ import annotations

import timm
import torch.nn as nn

# Logical arch name -> timm model name. ImageNet-1K, 224px, trained from scratch.
TIMM_NAMES = {
    "vit_b_16": "vit_base_patch16_224",
    "convnext_b": "convnext_base",
    "resnet50": "resnet50",
}


def build_classifier(
    arch: str,
    num_classes: int = 1000,
    pretrained: bool = False,
    drop_path_rate: float = 0.0,
    image_size: int = 224,
) -> nn.Module:
    """Build a supervised classifier backbone+head via timm.

    Args:
        arch: one of TIMM_NAMES keys (vit_b_16 | convnext_b | resnet50).
        num_classes: classifier head width (1000 for ImageNet-1K).
        pretrained: load timm pretrained weights (False = train from scratch).
        drop_path_rate: stochastic depth; standard for ViT/ConvNeXt recipes.
        image_size: input resolution; passed to ViT for pos-embed sizing.
    """
    if arch not in TIMM_NAMES:
        raise ValueError(f"Unknown supervised arch '{arch}'. Known: {sorted(TIMM_NAMES)}")

    kwargs = dict(pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path_rate)
    # resnet50 in timm does not accept img_size; ViT/ConvNeXt size is fixed by the variant at 224.
    return timm.create_model(TIMM_NAMES[arch], **kwargs)
