import math
import os
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist



class GatherFeatures(torch.autograd.Function):
    """
    Gather tensors from all processes, supporting differentiation.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.batch_size = x.shape[0]
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        # Reduce grads to the correct device/rank
        # Clip gradient based on batch size
        grad_size = grads[0].shape[0]
        # Ensure gradients are scaled correctly based on original batch size
        # Original code uses torch.distributed.get_rank() but PTL manages ranks
        local_rank = dist.get_rank()
        return all_gradients[local_rank] * ctx.batch_size / grad_size


gather_features = GatherFeatures.apply


class ClipLoss(nn.Module):
    """
    CLIP loss (InfoNCE) implementation.
    Handles local and distributed training.
    """
    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0, # Set by Lightning automatically
        world_size: int = 1, # Set by Lightning automatically
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # Cache for labels
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # Calculated labels
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor, # This is the learned parameter from the model
        output_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute CLIP InfoNCE loss.

        Args:
            image_features: Features for images [local_batch_size, feat_dim]
            text_features: Features for text [local_batch_size, feat_dim]
            logit_scale: Learned temperature parameter (scalar tensor)
            output_dict: Whether to return loss components as a dict

        Returns:
            Loss tensor or dict containing loss components.
        """
        device = image_features.device

        # --- Feature Gathering for Distributed Training ---
        if self.world_size > 1:
            # Gather features from all GPUs
            if self.gather_with_grad:
                # Allows backprop through gathered features
                all_image_features = gather_features(image_features)
                all_text_features = gather_features(text_features)
                if not self.local_loss:
                    # Concatenate features if not using local loss calculation
                    image_features = torch.cat(all_image_features, dim=0)
                    text_features = torch.cat(all_text_features, dim=0)
            else:
                 # Gather features without backprop support (more efficient)
                all_image_features = [torch.zeros_like(image_features) for _ in range(self.world_size)]
                all_text_features = [torch.zeros_like(text_features) for _ in range(self.world_size)]
                dist.all_gather(all_image_features, image_features)
                dist.all_gather(all_text_features, text_features)
                if not self.local_loss:
                    image_features = torch.cat(all_image_features, dim=0)
                    text_features = torch.cat(all_text_features, dim=0)

        # --- Logits Calculation ---
        # Features are assumed to be normalized already by the model encode methods
        # logit_scale is learnable parameter, apply exp()
        # Clamp logit scale to prevent instability (from OpenCLIP)
        logit_scale = torch.clamp(logit_scale.exp(), max=100)

        if self.local_loss:
            # Calculate local logits only
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        else:
             # Calculate global logits (all features vs all features)
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T # More efficient

        # --- Loss Calculation ---
        num_logits = logits_per_image.shape[0]
        labels = self.get_ground_truth(device, num_logits)

        loss_img = F.cross_entropy(logits_per_image, labels, reduction='mean')
        loss_txt = F.cross_entropy(logits_per_text, labels, reduction='mean')

        # Average the image and text loss
        loss = (loss_img + loss_txt) / 2
        
        if output_dict:
            return {
                "contrastive_loss": loss,
                "loss_img": loss_img,
                "loss_txt": loss_txt,
                "logits_per_image": logits_per_image,
                "logits_per_text": logits_per_text,
                "logit_scale": logit_scale # Return the processed scale
            }
        else:
            return loss 
