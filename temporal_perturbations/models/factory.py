import torch
from torch.nn import DataParallel
import timm
import wandb

from temporal_perturbations.models.clip_hba.clip_hba_utils import (
    CLIPHBA,
    apply_dora_to_ViT,
    switch_dora_layers,
)

from temporal_perturbations.data.spose_dimensions import classnames66

# Maps the user-facing architecture names to timm model identifiers
_TIMM_ARCH_MAP = {
    "ViT-B/16":   "vit_base_patch16_224",
    "ViT-L/14":   "vit_large_patch14_224",
    "RN50":       "resnet50",
    "RN101":      "resnet101",
    "ConvNeXt-B": "convnext_base",
}

def build_cliphba(
    backbone, vision_layers, transformer_layers, rank, 
    cuda, device, wandb_watch_model, wandb_log_freq
    ):
    """
    Initialize model and move to device.
    
    Args:
        config: Configuration dictionary containing model parameters
        device: Target device for model
        
    Returns:
        torch.nn.Module: Configured model ready for training
    """
    # Determine positional embedding based on backbone
    pos_embedding = False if backbone == 'RN50' else True
    print(f"pos_embedding is {pos_embedding}")
    
    # Initialize base model
    model = CLIPHBA(
        classnames=classnames66, 
        backbone_name=backbone, 
        pos_embedding=pos_embedding
    )
    
    # Apply DoRA adapters
    apply_dora_to_ViT(
        model, 
        n_vision_layers=vision_layers,
        n_transformer_layers=transformer_layers,
        r=rank,
        dora_dropout=0.1
    )
    switch_dora_layers(model, freeze_all=True, dora_state=True)
    
    # Use DataParallel if using all GPUs
    if cuda == -1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    model.to(device)

    if wandb_watch_model is True and wandb.run is not None:
        wandb.watch(model, log='all', log_freq=wandb_log_freq)
    
    return model


def build_model(
    architecture, 
    pretrained, 
    num_classes,
    clip_hba_backbone, 
    vision_layers, 
    transformer_layers, 
    rank, 
    cuda, 
    device, 
    wandb_watch_model, 
    wandb_log_freq
    ):
    
    if architecture == 'ViT-B/16':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)

    elif architecture == 'ViT-L/14':
        model = timm.create_model('vit_large_patch14_224', pretrained=pretrained, num_classes=num_classes)

    elif architecture == 'RN50':
        model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)

    elif architecture == 'ConvNeXt-B':
        model = timm.create_model('convnext_base', pretrained=pretrained, num_classes=num_classes)

    elif architecture == 'CLIP-HBA':
        model = build_cliphba(
            clip_hba_backbone,
            vision_layers,
            transformer_layers,
            rank,
            cuda,
            device,
            wandb_watch_model,
            wandb_log_freq
        )
        return model

    else:
        raise ValueError(f"Unknown architecture type: {architecture}")

    # Wrap with DataParallel when cuda=-1 (use all available GPUs)
    if cuda == -1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)

    return model