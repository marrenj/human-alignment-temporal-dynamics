"""CLIP-from-scratch model + Lightning checkpoint loader.

Loads the YFCC15M-trained CLIP Lightning checkpoints (under `runs_root`,
`clip_vitb32_*`) into the evaluation flow.
"""

from temporal_perturbations.models.clip_yfcc.ckpt_loader import (
    load_clip_lightning,
    load_clip_state_dict,
)
from temporal_perturbations.models.clip_yfcc.feature_space import (
    CLIP_FEATURES,
    DEFAULT_CLIP_FEATURE,
    load_clip_feature_model,
)
from temporal_perturbations.models.clip_yfcc.model import (
    CLIP,
    CLIPVisionCLSExtractor,
    CLIPVisionProjectedExtractor,
)

__all__ = [
    "CLIP",
    "CLIPVisionCLSExtractor",
    "CLIPVisionProjectedExtractor",
    "CLIP_FEATURES",
    "DEFAULT_CLIP_FEATURE",
    "load_clip_feature_model",
    "load_clip_lightning",
    "load_clip_state_dict",
]
