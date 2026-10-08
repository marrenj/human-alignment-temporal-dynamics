"""CLIP model backed by OpenCLIP.

Wraps open_clip.create_model() so the rest of the codebase sees the same API
(encode_image, encode_text, logit_scale, visual, tokenize, get_tokenizer).
Fixes the old hand-rolled model's architectural bugs: causal mask is now
applied in the text encoder, and text pooling uses EOT-token position.
"""

import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from temporal_perturbations.scratch_training.config import CLIPConfig, VIT_B_32_CONFIG


def _config_to_openclip_name(config: CLIPConfig) -> str:
    vision_cfg = config.vision_cfg
    if isinstance(vision_cfg.layers, list):
        return "RN50"
    if vision_cfg.patch_size == 32 and vision_cfg.width == 768:
        return "ViT-B-32"
    if vision_cfg.patch_size == 16 and vision_cfg.width == 768:
        return "ViT-B-16"
    raise ValueError(
        f"Cannot map config to OpenCLIP model: "
        f"patch_size={vision_cfg.patch_size}, width={vision_cfg.width}, "
        f"layers={vision_cfg.layers}"
    )


def remap_legacy_state_dict(state_dict: dict) -> dict:
    """Map keys from the old hand-rolled CLIP model to the OpenCLIP-backed wrapper.

    Call this before load_state_dict when loading checkpoints saved in the
    hand-rolled key format (e.g. those `models/clip_yfcc` reads).
    """
    remapped = {}
    for key, value in state_dict.items():
        new_key = key

        # Text encoder modules lived under 'text.' in the old model;
        # OpenCLIP puts them at the model root.
        if new_key.startswith("text."):
            new_key = new_key[len("text."):]

        # Transformer block indexing: transformer.{i} → transformer.resblocks.{i}
        new_key = re.sub(r"transformer\.(\d+)\.", r"transformer.resblocks.\1.", new_key)

        # LayerNorm naming
        new_key = new_key.replace(".norm1.", ".ln_1.")
        new_key = new_key.replace(".norm2.", ".ln_2.")

        # Attention: fused QKV → in_proj, output proj → out_proj
        new_key = new_key.replace(".attn.qkv.weight", ".attn.in_proj_weight")
        new_key = new_key.replace(".attn.qkv.bias", ".attn.in_proj_bias")
        new_key = new_key.replace(".attn.proj.weight", ".attn.out_proj.weight")
        new_key = new_key.replace(".attn.proj.bias", ".attn.out_proj.bias")

        # MLP: sequential indices → named layers
        new_key = new_key.replace(".mlp.0.", ".mlp.c_fc.")
        new_key = new_key.replace(".mlp.3.", ".mlp.c_proj.")

        # Wrapper prefix
        new_key = "_model." + new_key

        remapped[new_key] = value

    # logit_bias lives on the wrapper, not under _model
    if "_model.logit_bias" in remapped:
        remapped["logit_bias"] = remapped.pop("_model.logit_bias")

    return remapped


def is_legacy_state_dict(state_dict: dict) -> bool:
    """Detect whether a state_dict uses the old hand-rolled key format."""
    sample_keys = list(state_dict.keys())[:20]
    has_model_prefix = any(k.startswith("_model.") for k in sample_keys)
    has_old_text = any(k.startswith("text.") for k in sample_keys)
    has_old_norm = any(".norm1." in k or ".norm2." in k for k in state_dict)
    return not has_model_prefix and (has_old_text or has_old_norm)


class CLIP(nn.Module):
    """Thin wrapper around an OpenCLIP CLIP model."""

    def __init__(self, config: CLIPConfig = VIT_B_32_CONFIG):
        super().__init__()
        self.config = config
        self._openclip_name = _config_to_openclip_name(config)

        self._model = open_clip.create_model(self._openclip_name, pretrained=None)
        self._model.logit_scale = nn.Parameter(
            torch.ones([]) * config.init_logit_scale
        )

        if config.init_logit_bias is not None:
            self.logit_bias = nn.Parameter(
                torch.ones([]) * config.init_logit_bias
            )
        else:
            self.logit_bias = None

        self._tokenizer = open_clip.get_tokenizer(self._openclip_name)
        rank_zero_info(f"Created OpenCLIP model '{self._openclip_name}' (randomly initialized).")

    @property
    def visual(self):
        return self._model.visual

    @property
    def logit_scale(self):
        return self._model.logit_scale

    @logit_scale.setter
    def logit_scale(self, value):
        self._model.logit_scale = value

    def encode_image(self, image, normalize: bool = False):
        return self._model.encode_image(image, normalize=normalize)

    def encode_text(self, text, normalize: bool = False):
        return self._model.encode_text(text, normalize=normalize)

    def forward(self, image, text=None):
        image_features = self.encode_image(image)
        if text is not None:
            text_features = self.encode_text(text)
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T
            if self.logit_bias is not None:
                logits_per_image = logits_per_image + self.logit_bias
                logits_per_text = logits_per_text + self.logit_bias
            if self.config.output_dict:
                return {
                    "image_features": image_features,
                    "text_features": text_features,
                    "logits_per_image": logits_per_image,
                    "logits_per_text": logits_per_text,
                    "logit_scale": self.logit_scale,
                }
            return logits_per_image, logits_per_text, self.logit_scale
        if self.config.output_dict:
            return {"image_features": image_features}
        return image_features

    def get_tokenizer(self):
        return self._tokenizer

    def tokenize(self, texts, **kwargs):
        context_length = kwargs.pop("context_length", self.config.text_cfg.context_length)
        tokens = self._tokenizer(texts, context_length=context_length)
        return {"input_ids": tokens}

    def load_legacy_state_dict(self, state_dict, strict=False):
        """Load a state_dict produced by the old hand-rolled model."""
        return self.load_state_dict(remap_legacy_state_dict(state_dict), strict=strict)
