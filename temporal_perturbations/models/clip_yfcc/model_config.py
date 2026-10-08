from pydantic import BaseModel, Field
from typing import Optional, List, Union, Tuple, Dict, Any


class CLIPVisionConfig(BaseModel):
    # Vision Transformer specific fields
    layers: Union[int, List[int]] = 12  # Support both ViT (int) and ResNet (list)
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 32
    image_size: int = 224
    
    # Layer scale initial value
    ls_init_value: Optional[float] = None
    
    # Dropout settings
    patch_dropout: float = 0.0
    
    # Positional embedding
    pos_embed_type: str = "learnable"
    
    # Pooling and normalization
    no_ln_pre: bool = False
    final_ln_after_pool: bool = False
    pool_type: str = "tok"
    output_tokens: bool = False
    
    # ResNet specific fields (only used when layers is a list)
    heads: Optional[int] = None
    input_resolution: Optional[int] = None
    output_dim: Optional[int] = None
    use_sync_bn: Optional[bool] = None
    bn_group_size: Optional[int] = None
    bn_sync_stats: Optional[bool] = None


class CLIPTextConfig(BaseModel):
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    
    # Layer scale initial value
    ls_init_value: Optional[float] = None
    
    # Text specific options
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False
    
    # Pooling and normalization
    final_ln_after_pool: bool = False
    pool_type: str = "argmax"
    
    # Projection settings
    proj_bias: bool = False
    proj_type: str = "linear"
    output_tokens: bool = False
    
    # Model name
    model_name: str = "openai/clip-vit-base-patch32"


class CLIPConfig(BaseModel):
    embed_dim: int = 512
    vision_cfg: CLIPVisionConfig = Field(default_factory=CLIPVisionConfig)
    text_cfg: CLIPTextConfig = Field(default_factory=CLIPTextConfig)
    
    # Additional settings
    quick_gelu: bool = False
    init_logit_scale: float = 4.60517  # ln(100)
    init_logit_bias: Optional[float] = None
    
    # Training settings
    cast_dtype: Optional[str] = None
    output_dict: bool = False
    text_model_name: str = "openai/clip-vit-base-patch32"
    text_pretrained: bool = True
    text_output_dim: int = 512
