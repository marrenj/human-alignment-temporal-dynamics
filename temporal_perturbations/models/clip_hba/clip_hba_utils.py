import os
from types import NoneType
import torch
from temporal_perturbations.models.clip_hba import clip
from pathlib import Path
import torch.nn as nn
import math
from torch.nn import functional as F
import tqdm


def load_clip_to_cpu(backbone_name):
    """
    Download (if necessary) and load a CLIP backbone onto CPU memory.

    Parameters
    ----------
    backbone_name : str
        Key referencing the desired CLIP checkpoint in ``clip._MODELS``.

    Returns
    -------
    torch.nn.Module
        The instantiated CLIP model in evaluation mode on the CPU.
    """
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CLIPHBA(nn.Module):
    """
    Thin wrapper around a CLIP backbone that freezes all weights and caches
    class prompts so CLIP-HBA (Human Behavior Alignment) scoring can be run
    efficiently for a fixed vocabulary.

    Parameters
    ----------
    classnames : list[str]
        Text prompts whose embeddings define the classifier heads.
    backbone_name : str, default 'RN50'
        Identifier for the CLIP checkpoint to load.
    pos_embedding : bool, default False
        Whether to use positional embeddings when forwarding images/prompts.
    """
    def __init__(self, classnames, backbone_name='RN50', pos_embedding=False):
        super().__init__()

        self.num_clip = len(classnames)
        self.clip_model = load_clip_to_cpu(backbone_name)
        self.clip_model.float()
        self.pos_embedding = pos_embedding

        # Disable gradients for all parameters first
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Tokenize all prompts at once and store them as a tensor
        self.tokenized_prompts = torch.stack([clip.tokenize(classname) for classname in classnames])
        self._tokenized_prompts_device = None  # Cache for device


    def forward(self, image):
        if self.clip_model.training:
            self.clip_model.eval()

        # Process all tokenized prompts in a single forward pass
        if self._tokenized_prompts_device != image.device:
            self.tokenized_prompts = self.tokenized_prompts.to(image.device)
            self._tokenized_prompts_device = image.device

        pred_score = self.clip_model(image, self.tokenized_prompts, self.pos_embedding)

        return pred_score.float()

    # CLIP-style adapter API: thin pass-throughs to the underlying CLIP
    # backbone (zero-shot classification / retrieval interface).

    def encode_image(self, image, normalize: bool = True):
        """Image branch of the CLIP backbone. Returns L2-normalized features by default."""
        feats = self.clip_model.encode_image(image, self.pos_embedding)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, text, normalize: bool = True):
        """Text branch of the CLIP backbone. Returns L2-normalized features by default."""
        feats = self.clip_model.encode_text(text)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @property
    def logit_scale(self):
        """Expose the learned temperature parameter from the CLIP backbone."""
        return self.clip_model.logit_scale

    @property
    def visual(self):
        """Expose the visual tower so callers can read its output dim / dtype."""
        return self.clip_model.visual

    @staticmethod
    def get_tokenizer():
        """Return a callable tokenizer with the open_clip-style interface.

        clip.tokenize already returns shape [N, 77] for a list of N strings.
        """
        def _tokenize(texts):
            if isinstance(texts, str):
                texts = [texts]
            return clip.tokenize(list(texts))
        return _tokenize


class DoRALayer(nn.Module):
    """
    Directional Rank-One Adaptation (DoRA) wrapper that freezes the original
    linear layer weights into magnitude ``m`` and direction ``D`` components,
    then learns a low-rank LoRA-style update to the direction while keeping a
    trainable magnitude scaling. Optionally applies dropout to the adaptation.

    Parameters
    ----------
    original_layer : nn.Linear
        Pretrained linear layer whose weights/bias are copied as the baseline.
    r : int, default 8
        Rank of the low-rank adaptation matrices ``delta_D_A`` and ``delta_D_B``.
    dora_alpha : float, default 16
        Scaling applied to the low-rank update (similar to LoRA alpha / r).
    dora_dropout : float, default 0.1
        Dropout probability applied to the low-rank delta before recomposing
        the direction matrix.
    """
    def __init__(self, original_layer, r=8, dora_alpha=16, dora_dropout=0.1):
        super(DoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r  # Low-rank factor
        self.dora_alpha = dora_alpha  # Scaling parameter
        self.dora_dropout = nn.Dropout(p=dora_dropout)

        # Decompose original weights into magnitude and direction
        with torch.no_grad():
            W = original_layer.weight.data.clone()  # [out_features, in_features]
            W = W.T  # Transpose to [in_features, out_features]
            S = torch.norm(W, dim=0)  # Magnitudes (norms of columns), shape [out_features]
            D = W / S  # Direction matrix with unit-norm columns, shape [in_features, out_features]

        # Store S as a trainable parameter
        self.m = nn.Parameter(S)  # [out_features]
        # Store D as a buffer (since we don't want to update it directly)
        self.register_buffer('D', D)  # [in_features, out_features]

        # LoRA adaptation of D
        self.delta_D_A = nn.Parameter(torch.zeros(self.r, original_layer.out_features))
        self.delta_D_B = nn.Parameter(torch.zeros(original_layer.in_features, self.r))

        # Scaling
        self.scaling = self.dora_alpha / self.r

        # Initialize delta_D_A and delta_D_B
        self.reset_parameters()

        # Copy the bias from the original layer
        if self.original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.delta_D_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.delta_D_B, a=math.sqrt(5))

    @property
    def weight(self):
        # Compute adapted D
        delta_D = (self.delta_D_B @ self.delta_D_A) * self.scaling  # [in_features, out_features]

        D_new = self.D + delta_D  # [in_features, out_features]

        # Normalize columns of D_new
        D_norms = torch.norm(D_new, dim=0, keepdim=True) + 1e-8  # [1, out_features], add epsilon to avoid division by zero
        D_normalized = D_new / D_norms  # [in_features, out_features]

        # Reconstruct the adapted weight
        W = D_normalized * self.m  # [in_features, out_features], m is [out_features]

        W = W.T  # Transpose back to [out_features, in_features]

        return W

    def forward(self, x):
        # Compute adapted D
        delta_D = (self.delta_D_B @ self.delta_D_A) * self.scaling  # [in_features, out_features]
        delta_D = self.dora_dropout(delta_D)

        D_new = self.D + delta_D  # [in_features, out_features]

        # Normalize columns of D_new
        D_norms = torch.norm(D_new, dim=0, keepdim=True) + 1e-8  # [1, out_features]
        D_normalized = D_new / D_norms  # [in_features, out_features]

        # Reconstruct the adapted weight
        W = D_normalized * self.m  # [in_features, out_features]
        W = W.T  # [out_features, in_features]

        # Compute output
        return F.linear(x, W, self.bias)


def apply_dora_to_ViT(model, n_vision_layers=1, n_transformer_layers=1, r=8, dora_dropout=0.1):
    """
    Wrap the last ``n`` attention output projections of both the visual and text
    transformers in a CLIP-HBA model with `DoRALayer`s.

    Parameters
    ----------
    model : nn.Module | nn.DataParallel
        CLIP-HBA model whose ``clip_model.visual`` and ``clip_model.transformer``
        blocks will be modified in-place.
    n_vision_layers : int, default 1
        Number of final visual transformer residual blocks to adapt.
    n_transformer_layers : int, default 1
        Number of final text transformer residual blocks to adapt.
    r : int, default 8
        Rank used when constructing each ``DoRALayer``.
    dora_dropout : float, default 0.1
        Dropout probability applied inside each DoRA adapter.
    seed : int, default 123
        Unused placeholder for future deterministic init hooks (retained for API
        compatibility).
    """
    
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Specific blocks to modify in the visual transformer
    block_indices = range(-n_vision_layers, 0)  # Adjusted for proper indexing

    for idx in block_indices:
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.visual.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a DoRALayer
        dora_layer = DoRALayer(target_layer, r=r, dora_dropout=dora_dropout)
        target_block.attn.out_proj = dora_layer

    # Specific blocks to modify in the main transformer
    block_indices = range(-n_transformer_layers, 0)

    for idx in block_indices:
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a DoRALayer
        dora_layer = DoRALayer(target_layer, r=r, dora_dropout=dora_dropout)
        target_block.attn.out_proj = dora_layer


def switch_dora_layers(model, freeze_all=True, dora_state=True):
    """
    Freeze or unfreeze the model's parameters based on the presence of DoRA layers.
    If a DoRALayer is encountered, only its specific DoRA parameters are unfrozen.
    """
    for name, param in model.named_parameters():
        # Initially set requires_grad based on the freeze_all flag
        param.requires_grad = not freeze_all

    if freeze_all:
        # If freezing all parameters, selectively unfreeze DoRA parameters
        def recursive_unfreeze_dora(module):
            for child_name, child in module.named_children():
                if isinstance(child, DoRALayer):
                    # Unfreeze DoRA-specific parameters within DoRALayer
                    child.m.requires_grad = dora_state
                    child.delta_D_A.requires_grad = dora_state
                    child.delta_D_B.requires_grad = dora_state
                    # Keep the original layer's parameters frozen
                    if child.bias is not None:
                        child.bias.requires_grad = False
                else:
                    recursive_unfreeze_dora(child)

        # Apply selective unfreezing to the entire model
        if isinstance(model, torch.nn.DataParallel):
            recursive_unfreeze_dora(model.module)
        else:
            recursive_unfreeze_dora(model)


def evaluate_model(model, data_loader, device, criterion):
    """
    Run a full evaluation pass computing the dataset-wide average loss.

    Parameters
    ----------
    model : nn.Module
        Model whose ``forward`` method produces predictions from batched images.
    data_loader : torch.utils.data.DataLoader
        Iterable yielding tuples of (metadata, images, targets).
    device : torch.device | str
        Device onto which images/targets are moved before inference.
    criterion : Callable
        Loss function accepting ``(predictions, targets)``.

    Returns
    -------
    float
        Mean loss over every sample in ``data_loader.dataset``.
    """
    model.eval()
    total_loss = 0.0

    # Wrap data_loader with tqdm for a progress bar
    with torch.no_grad(), tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating") as progress_bar:
        for batch_idx, (_, images, targets) in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            total_loss += loss.item() * images.size(0) 

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def save_dora_parameters(
    model,
    dora_parameters_path,
    epoch,
    vision_layers,
    transformer_layers,
    log_fn=None,
):
    """
    Save DoRA parameters for the final ``vision_layers`` visual blocks and the
    final ``transformer_layers`` text blocks.

    Parameters
    ----------
    model : nn.Module | nn.DataParallel
        Model containing the adapted CLIP backbone.
    dora_parameters_path : str | Path
        Directory where the serialized parameters will be written.
    epoch : int
        Epoch index used when naming the checkpoint.
    vision_layers : int
        Number of final visual transformer blocks instrumented with DoRA.
    transformer_layers : int
        Number of final text transformer blocks instrumented with DoRA.
    log_fn : Callable[[str], None], optional
        Logging function (e.g., ``logger.info``); when ``None`` the function
        remains silent.
    """

    def _resolve(attr_path):
        module = model_module
        for attr in attr_path.split("."):
            module = getattr(module, attr)
        return module

    def _build_specs(block_path, count):
        if not count:
            return []
        blocks = _resolve(block_path)
        total = len(blocks)
        if count > total:
            raise ValueError(
                f"Requested {count} blocks for '{block_path}' but only {total} exist."
            )
        start_idx = total - count
        return [
            f"{block_path}.{idx}.attn.out_proj"
            for idx in range(start_idx, total)
        ]

    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model

    modules_to_save = []
    modules_to_save.extend(
        _build_specs(
            "clip_model.visual.transformer.resblocks",
            vision_layers,
        )
    )
    modules_to_save.extend(
        _build_specs(
            "clip_model.transformer.resblocks",
            transformer_layers,
        )
    )

    if not modules_to_save:
        if log_fn:
            log_fn("No DoRA modules selected for saving.")
        return

    dora_params = {}
    for module_path in modules_to_save:
        module = model_module
        for attr in module_path.split("."):
            module = getattr(module, attr)

        dora_params[f"{module_path}.m"] = module.m.detach().cpu()
        dora_params[f"{module_path}.delta_D_A"] = module.delta_D_A.detach().cpu()
        dora_params[f"{module_path}.delta_D_B"] = module.delta_D_B.detach().cpu()

    os.makedirs(dora_parameters_path, exist_ok=True)
    save_path = os.path.join(dora_parameters_path, f"epoch{epoch}_dora_params.pth")
    torch.save(dora_params, save_path)


def dora_parameter_keys(model):
    """State-dict keys of every DoRA adapter parameter in ``model`` (``DataParallel`` unwrapped)."""
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    return {
        f"{name}.{param}"
        for name, child in module.named_modules() if isinstance(child, DoRALayer)
        for param in ("m", "delta_D_A", "delta_D_B")
    }


def load_dora_state_dict(model, state_dict, source="state_dict"):
    """Load DoRA adapter tensors into ``model``; raise unless the checkpoint covers the adapters exactly."""
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    expected = dora_parameter_keys(module)
    if not expected:
        raise RuntimeError(f"{source}: model has no DoRA layers to load into")
    unexpected = sorted(set(state_dict) - expected)
    missing = sorted(expected - set(state_dict))
    if unexpected or missing:
        raise RuntimeError(
            f"{source}: DoRA checkpoint does not match the model's adapters "
            f"(unexpected keys: {unexpected[:5]}, missing keys: {missing[:5]}); "
            "check vision_layers / transformer_layers / rank against the run's resolved_config.yaml"
        )
    # strict=False only tolerates the frozen backbone keys a DoRA checkpoint never holds;
    # shape mismatches still raise inside load_state_dict.
    module.load_state_dict(state_dict, strict=False)


def load_dora_checkpoint(model, checkpoint_root, epoch, *, dora_dir="dora_params", map_location="cpu"):
    """Load ``epoch{epoch}_dora_params.pth`` from ``checkpoint_root/dora_dir`` into ``model``; return its path."""
    checkpoint_root = Path(checkpoint_root)
    dora_path = checkpoint_root / dora_dir if dora_dir is not None else checkpoint_root
    checkpoint_path = dora_path / f"epoch{epoch}_dora_params.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing DoRA checkpoint: {checkpoint_path}")
    load_dora_state_dict(
        model,
        torch.load(checkpoint_path, map_location=map_location, weights_only=True),
        source=str(checkpoint_path),
    )
    return checkpoint_path


def initialize_cliphba_model(
    backbone_name,
    classnames,
    vision_layers,
    transformer_layers,
    rank,
    dora_dropout,
    logger=None
):

    
    # Determine pos_embedding based on backbone
    pos_embedding = False if backbone_name == 'RN50' else True
    logger.info(f"pos_embedding is {pos_embedding}")

    model = CLIPHBA(classnames=classnames, backbone_name=backbone_name, 
                pos_embedding=pos_embedding)
    model.eval() # inference mode

    # Apply DoRA
    apply_dora_to_ViT(model, 
                      n_vision_layers=vision_layers,
                      n_transformer_layers=transformer_layers,
                      r=rank,
                      dora_dropout=dora_dropout)
    
    return model