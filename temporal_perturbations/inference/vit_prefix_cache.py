"""Split a CLIP-HBA ViT forward pass into a frozen prefix and a per-checkpoint suffix.

`apply_dora_to_ViT` (clip_hba_utils.py) only wraps the LAST `n_vision_layers`
visual resblocks (and last `n_transformer_layers` text block) in DoRA
adapters; every earlier block keeps its original pretrained-CLIP weights,
which are set once in `initialize_cliphba_model` and never touched again
(`load_dora_checkpoint` uses `strict=False` and only overwrites DoRA
parameters). So for a fixed (backbone, vision_layers, transformer_layers),
this prefix -- patch embed through resblocks[:split_at] -- is bit-identical
across EVERY checkpoint that shares those settings: every epoch of a run,
and every run (baseline, any perturbation, any seed). It only needs to be
computed once per stimulus set, no matter how many checkpoints get compared
against it.

Used by `run_things_brain_rsa_cached.py` (same run, many epochs) and
`run_model_to_model_rsa.py` (many runs, many epochs).
"""

import torch


def visual_prefix(visual, images, pos_embedding, split_at):
    """Replicate VisionTransformer.forward up to resblocks[:split_at]. Returns LND."""
    x = visual.conv1(images.type(visual.conv1.weight.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    if pos_embedding:
        pe = torch.nn.functional.interpolate(
            visual.positional_embedding.unsqueeze(0).unsqueeze(0),
            size=(x.size(1), x.size(2)), mode="bicubic").squeeze(0).squeeze(0)
        x = x + pe.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    for blk in visual.transformer.resblocks[:split_at]:
        x = blk(x)
    return x


def visual_suffix(visual, prefix_lnd, split_at):
    """Replicate the remaining VisionTransformer.forward from resblocks[split_at:]."""
    x = prefix_lnd
    for blk in visual.transformer.resblocks[split_at:]:
        x = blk(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        x = x @ visual.proj
    return x


def text_features(clip_model, tokenized_prompts):
    """Encode the class/SPoSE prompts and L2-normalize (matches CLIP.forward)."""
    feats = torch.stack([clip_model.encode_text(p) for p in tokenized_prompts])
    feats = feats.view(-1, feats.size(-1))
    return feats / feats.norm(dim=-1, keepdim=True)
