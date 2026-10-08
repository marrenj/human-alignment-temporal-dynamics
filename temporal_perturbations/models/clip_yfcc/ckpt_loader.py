"""Lightning checkpoint loader for the CLIP ViT-B/32 (YFCC15M) runs.

The checkpoints pickle references to the trainer's top-level `config.*`
dataclasses alongside the weights, so a bare `torch.load` fails with
`ModuleNotFoundError: No module named 'config'`. Loading goes through
`models.lightning_ckpt.load_lightning_state_dict`, whose permissive unpickler
skips those references.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn


def load_clip_state_dict(ckpt_path: str | Path, map_location="cpu") -> dict:
    """Load just the CLIP state_dict from a Lightning checkpoint.

    Returns the inner-model state_dict (with the `model.` prefix stripped),
    suitable for `clip_model.load_state_dict(...)`.

    Parameters
    ----------
    ckpt_path : str | Path
    map_location : str | torch.device
        Forwarded to torch.load.

    Returns
    -------
    dict[str, torch.Tensor]
    """
    # The permissive unpickler maps any unresolvable pickled config class to a
    # throwaway placeholder, so the checkpoint's `config.*` hyper-parameter
    # references never need to be importable.
    from temporal_perturbations.models.lightning_ckpt import load_lightning_state_dict

    return load_lightning_state_dict(ckpt_path, map_location=map_location)


def load_clip_lightning(
    ckpt_path: str | Path,
    *,
    config_preset: str = "vit_b_32",
    map_location="cpu",
) -> nn.Module:
    """Build the ``config_preset`` CLIP and load ``ckpt_path`` into it with ``strict=True``."""
    from temporal_perturbations.models.clip_yfcc.model import CLIP
    from temporal_perturbations.models.clip_yfcc.model_loader import load_model_config

    state_dict = load_clip_state_dict(ckpt_path, map_location=map_location)
    cfg = load_model_config(config_preset)

    model = CLIP(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
