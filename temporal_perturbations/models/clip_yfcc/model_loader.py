"""Minimal model-config loader: just the architecture loading evaluation needs (no training-config or wandb plumbing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from temporal_perturbations.models.clip_yfcc.model_config import (
    CLIPConfig,
    CLIPVisionConfig,
    CLIPTextConfig,
)

MODELCONFIG_DIR = Path(__file__).parent / "modelconfig"


def _load_json(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def load_model_config(model_name: str) -> CLIPConfig:
    """Read modelconfig/<model_name>.json and assemble a CLIPConfig."""
    config_data = _load_json(MODELCONFIG_DIR / f"{model_name}.json")

    vision_cfg = CLIPVisionConfig(**config_data["vision_cfg"])
    text_cfg = CLIPTextConfig(**config_data["text_cfg"])

    rest = dict(config_data)
    rest.pop("vision_cfg", None)
    rest.pop("text_cfg", None)
    rest.pop("name", None)

    return CLIPConfig(vision_cfg=vision_cfg, text_cfg=text_cfg, **rest)


def get_vit_b_32_config() -> CLIPConfig:
    return load_model_config("vit_b_32")


def get_resnet_50_config() -> CLIPConfig:
    return load_model_config("resnet_50")


# Module-level constants (referenced by model.py and a few callers).
VIT_B_32_CONFIG = get_vit_b_32_config()
RESNET_50_CONFIG = get_resnet_50_config()
