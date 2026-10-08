import math
import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, CLIPTextModelWithProjection
# Soft-import: rank_zero_info is only used for INFO-level construction logging.
# Eval-only environments don't need pytorch_lightning.
try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_info
except ImportError:
    def rank_zero_info(*args, **kwargs):  # type: ignore[no-redef]
        pass

from temporal_perturbations.models.clip_yfcc.model_config import CLIPConfig
from temporal_perturbations.models.clip_yfcc.model_loader import VIT_B_32_CONFIG
from temporal_perturbations.models.clip_yfcc.resnet import build_resnet_50


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        return super().forward(x.type(torch.float32)).type(orig_type)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        scaled_cosine: bool = False,
        scale_heads: bool = False,
        logit_scale_max: float = math.log(1. / 0.01),
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # F.scaled_dot_product_attention has no hook for scaled-cosine logits or per-head scaling, so both are disabled when it is available.
        if hasattr(F, 'scaled_dot_product_attention'):
            if scaled_cosine:
                print("Warning: scaled_cosine is incompatible with PyTorch's F.scaled_dot_product_attention. Ignoring scaled_cosine.")
                self.scaled_cosine = False
            else:
                self.scaled_cosine = scaled_cosine

            if scale_heads:
                 print("Warning: scale_heads is incompatible with PyTorch's F.scaled_dot_product_attention. Ignoring scale_heads.")
                 self.scale_heads = False
            else:
                 self.scale_heads = scale_heads
        else:
            self.scaled_cosine = scaled_cosine
            if self.scaled_cosine:
                self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
                self.logit_scale_max = logit_scale_max
                self.scale_heads = scale_heads
            else:
                self.scale_heads = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if hasattr(F, 'scaled_dot_product_attention'):
            # is_causal=False for bidirectional attention used in ViT encoders
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )
            x = x.transpose(1, 2).reshape(B, N, C) # (B, N, num_heads * head_dim)

        else:
            if self.scaled_cosine:
                attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
                logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
                attn = attn * logit_scale
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            if self.scale_heads:
                x = x.reshape(B, N, self.num_heads, self.head_dim)
                x = x * self.scale.reshape(1, 1, -1, 1)
                x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        ls_init_value: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.ls_init_value = ls_init_value
        if ls_init_value is not None:
            self.gamma_1 = nn.Parameter(ls_init_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(ls_init_value * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None
            
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False):
        if use_checkpoint and self.training:
            # Use gradient checkpointing for memory efficiency
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x: torch.Tensor):
        if self.gamma_1 is None:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.gamma_1 * self.attn(self.norm1(x))
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 32,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        output_dim: int = 512,
        ls_init_value: Optional[float] = None,
        patch_dropout: float = 0.,
        no_ln_pre: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        pos_embed_type: str = 'learnable'
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.output_dim = output_dim
        
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        self.class_embedding = nn.Parameter(torch.randn(width))
        
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(torch.randn(self.num_patches + 1, width) * 0.01)
        else:
            raise NotImplementedError(f"Position embedding type {pos_embed_type} not implemented")
            
        if no_ln_pre:
            self.ln_pre = nn.Identity()
        else:
            self.ln_pre = norm_layer(width)
        
        self.patch_dropout = patch_dropout
        self.patch_drop = nn.Dropout(patch_dropout)
        
        self.transformer = nn.ModuleList([
            TransformerBlock(
                dim=width,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for _ in range(layers)
        ])
        
        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(torch.randn(width, output_dim) * (1 / width ** 0.5))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # Calculate scale factors similar to DECLIP
        proj_std = (self.width ** -0.5) * ((2 * len(self.transformer)) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        
        for block in self.transformer:
            nn.init.normal_(block.attn.qkv.weight, std=attn_std)
            nn.init.normal_(block.attn.proj.weight, std=proj_std)
            
            # MLP weights - in your implementation, these are at indices 0 and 3
            nn.init.normal_(block.mlp[0].weight, std=fc_std)  # First linear layer
            nn.init.normal_(block.mlp[3].weight, std=proj_std)  # Second linear layer
        
        if self.proj is not None:
            nn.init.normal_(self.proj, std=self.width ** -0.5)

    def forward(self, x: torch.Tensor, return_cls_token: bool = False):
        x = self.conv1(x)  # [batch, width, grid_h, grid_w]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, width, grid_h * grid_w]
        x = x.permute(0, 2, 1)  # [batch, grid_h * grid_w, width]

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)  # [batch, grid_h * grid_w + 1, width]

        x = x + self.positional_embedding.to(x.dtype)

        if self.patch_dropout > 0 and self.training:
            # Keep class token and drop patches
            x = torch.cat([
                x[:, :1, :],
                self.patch_drop(x[:, 1:, :])
            ], dim=1)

        x = self.ln_pre(x)

        for block in self.transformer:
            x = block(x)

        x = x[:, 0]

        x = self.ln_post(x)

        # `return_cls_token=True` stops here: the vision encoder's own CLS
        # token embedding, before the linear projection into the joint
        # image-text space. Used by evaluations that want
        # the encoder's native visual representation rather than the
        # contrastive-projected one.
        if return_cls_token:
            return x

        x = x @ self.proj

        return x


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: Optional[float] = None,
        output_dim: int = 512,
        pad_id: int = 0,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        embed_cls: bool = False,
        no_causal_mask: bool = False,
        model_name: str = "openai/clip-vit-base-patch32",  # tokenizer source
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        
        self.token_embedding = nn.Embedding(vocab_size, width)
        
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        
        self.no_causal_mask = no_causal_mask
        self.embed_cls = embed_cls
        
        self.transformer = nn.ModuleList([
            TransformerBlock(
                dim=width,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for _ in range(layers)
        ])
        
        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        self.pad_id = pad_id
        
        self.initialize_parameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # Calculate scale factors similar to DECLIP
        proj_std = (self.width ** -0.5) * ((2 * len(self.transformer)) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        
        for block in self.transformer:
            nn.init.normal_(block.attn.qkv.weight, std=attn_std)
            nn.init.normal_(block.attn.proj.weight, std=proj_std)
            
            # MLP weights - in your implementation, these are at indices 0 and 3
            nn.init.normal_(block.mlp[0].weight, std=fc_std)  # First linear layer
            nn.init.normal_(block.mlp[3].weight, std=proj_std)  # Second linear layer
        
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width ** -0.5)

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # Zero out the lower diagonal
        return mask

    def forward(self, text):  
        if isinstance(text, list) and isinstance(text[0], str):
            text = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.context_length
            )["input_ids"].to(self.text_projection.device)

        elif not isinstance(text, torch.Tensor):
            raise ValueError(f"Unsupported input type: {type(text)}. Expected list of str or tensor.")

        # Text shape: [batch_size, seq_len]
        x = self.token_embedding(text)  # [batch_size, seq_len, dim]
        
        x = x + self.positional_embedding
        
        if not self.no_causal_mask:
            causal_mask = self._build_causal_attention_mask(
                text.size(0), text.size(1), x.dtype
            ).to(x.device)
        else:
            causal_mask = None
        
        for block in self.transformer:
            x = block(x)
        
        x = self.ln_final(x)
        
        input_mask = (text != self.pad_id).unsqueeze(-1).to(x.dtype)
        
        if self.embed_cls:
            x = x[:, 0] @ self.text_projection
        else:
            x = (x * input_mask).sum(dim=1) / input_mask.sum(dim=1)
            x = x @ self.text_projection
            
        return x
    
class CLIP(nn.Module):
    def __init__(
        self,
        config: CLIPConfig = VIT_B_32_CONFIG, random_text_encoder: bool = False):
        super().__init__()
        self.config = config
        self._tokenizer = None

        vision_cfg = config.vision_cfg
        
        if hasattr(vision_cfg, 'layers') and isinstance(vision_cfg.layers, list):
            self.visual = build_resnet_50(
                layers=vision_cfg.layers,
                heads=vision_cfg.heads,
                input_resolution=vision_cfg.input_resolution,
                width=vision_cfg.width,
                output_dim=config.embed_dim
            )
            rank_zero_info("Using ResNet-50 vision encoder (randomly initialized or loaded from checkpoint).")
        else:
            self.visual = VisionTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_cfg.width // vision_cfg.head_width,
                mlp_ratio=vision_cfg.mlp_ratio,
                output_dim=config.embed_dim,
                ls_init_value=vision_cfg.ls_init_value,
                patch_dropout=vision_cfg.patch_dropout,
                no_ln_pre=vision_cfg.no_ln_pre,
                pos_embed_type=vision_cfg.pos_embed_type,
            )
            self.visual.initialize_parameters()
            rank_zero_info("Using custom VisionTransformer (randomly initialized or loaded from checkpoint).")

        text_cfg = config.text_cfg
        self.text = TextTransformer(
                context_length=text_cfg.context_length,
                vocab_size=text_cfg.vocab_size,
                width=text_cfg.width,
                heads=text_cfg.heads,
                layers=text_cfg.layers,
                ls_init_value=text_cfg.ls_init_value,
                output_dim=config.embed_dim,
                pad_id=text_cfg.pad_id,
                mlp_ratio=text_cfg.mlp_ratio,
                embed_cls=text_cfg.embed_cls,
                no_causal_mask=text_cfg.no_causal_mask,
            )
        self.text.initialize_parameters()
        rank_zero_info("Using custom TextTransformer (randomly initialized or loaded from checkpoint).")
        self.logit_scale = nn.Parameter(torch.ones([]) * config.init_logit_scale)
        if config.init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * config.init_logit_bias)
        else:
            self.logit_bias = None
        self._tokenizer = self.get_tokenizer()

    def encode_image(self, image, normalize: bool = False, return_cls_token: bool = False):
        features = self.visual(image, return_cls_token=return_cls_token)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features
    
    def encode_text(self, text, normalize: bool = False):
        if self.text is None:
            raise ValueError("Text encoder is not initialized")

        features = self.text(text)

        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    
    def forward(self, image, text=None):
        image_features = self.encode_image(image)
        
        if text is not None and self.text is not None:
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
                    "logit_scale": self.logit_scale
                }
            else:
                return logits_per_image, logits_per_text, self.logit_scale
        else:
            if self.config.output_dict:
                return {"image_features": image_features}
            else:
                return image_features

    def get_tokenizer(self):
        """Safely returns the tokenizer instance if available."""
        if self._tokenizer is None:
             # Attempt to initialize on demand if not done in init (e.g., if init failed silently)
             rank_zero_info("Attempting lazy initialization of tokenizer...")
             try:
                 cfg_model_name = getattr(getattr(self.config, 'text_cfg', None), 'model_name', None) \
                                  or getattr(self.config, 'text_model_name', None) # Try both places
                 if cfg_model_name:
                     self._tokenizer = AutoTokenizer.from_pretrained(cfg_model_name)
                     rank_zero_info(f"Lazy initialization of tokenizer '{cfg_model_name}' successful.")
                 else:
                      rank_zero_info("ERROR: Could not find tokenizer name in config for lazy initialization.")
             except Exception as e:
                 rank_zero_info(f"ERROR: Lazy initialization of tokenizer failed: {e}")
        return self._tokenizer

    def tokenize(self, texts, **kwargs):
        # Default kwargs for standard tokenization, can be overridden
        context_length = kwargs.pop("context_length", self.config.text_cfg.context_length)
        truncate = kwargs.pop("truncate", True)

        if self._tokenizer is None:
            raise ValueError("Cannot tokenize text because tokenizer is not available.")

        return self._tokenizer(
            texts,
            padding='max_length',
            truncation=truncate,
            max_length=context_length,
            return_tensors='pt',
            **kwargs, # Pass any remaining kwargs
        )


class CLIPVisionCLSExtractor(nn.Module):
    """Wraps a `CLIP` model so `forward(image)` returns the vision encoder's
    CLS token embedding (post `ln_post`, pre-projection) instead of the
    joint image-text embedding.

    For representational-geometry evaluations the contrastive
    projection head is trained to align images with captions, not to
    preserve visual similarity structure — comparing in the vision
    encoder's own CLS-token space (its native representation before that
    projection) is the standard choice for this kind of eval, matching the
    "CLS token" convention already used for CLIP layers in
    `frrsa_core.select_layers`.
    """

    def __init__(self, clip_model: "CLIP"):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(image, return_cls_token=True)


class CLIPVisionProjectedExtractor(nn.Module):
    """Wraps a `CLIP` model so `forward(image)` returns the L2-normalized joint
    image-text embedding — the CLS token *after* `x @ self.proj` (512-d for
    ViT-B/32), i.e. what `encode_image(..., normalize=True)` produces.

    This is the counterpart to `CLIPVisionCLSExtractor`, and the two differ in
    more than dimensionality: `proj` is the contrastive head trained by the
    InfoNCE loss to map images into the space text embeddings occupy, so this
    representation carries the caption-alignment geometry while the CLS-token
    one does not.

    Prefer `CLIPVisionCLSExtractor` for representational analyses (RSA / CKA /
    linear probes), which is why it is the default in those sweeps. This class
    exists so the projected space stays reproducible — it is the representation
    used for zero-shot / retrieval, and it is what the cross-architecture
    model-to-model sweep scored before the CLS default was adopted.
    """

    def __init__(self, clip_model: "CLIP"):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(image, normalize=True)
