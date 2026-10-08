"""Which representation to read out of a CLIP visual tower, and how to load it.

CLIP's vision encoder exposes two usable outputs, and they are NOT
interchangeable — they differ in dimensionality, in normalization, and in what
they encode:

  * ``cls``        — the CLS token after ``ln_post`` but BEFORE ``@ self.proj``
                     (768-d for ViT-B/32, unnormalized). The encoder's own
                     visual representation, and the analog of a timm model's
                     pooled penultimate feature.
  * ``projected``  — the joint image-text embedding, AFTER the projection and
                     L2-normalized (512-d for ViT-B/32). This is the zero-shot /
                     retrieval space.

``proj`` is the contrastive head trained by InfoNCE to align images with
captions, not to preserve visual similarity structure. Any analysis of
representational geometry — RSA, CKA, linear probes — therefore wants ``cls``,
which is why it is the default here and the convention the FR-RSA
(``frrsa_core.select_layers``) already follows for this checkpoint family. ``projected`` stays available so the retrieval space remains
reproducible.

These constants live in the library rather than in any one sweep script because
three evaluation drivers now select between the two spaces and must agree on the
names and semantics.
"""

from __future__ import annotations

CLIP_FEATURES = {
    "cls": "vision-encoder CLS token, post-ln_post, PRE-projection (768-d, unnormalized)",
    "projected": "joint image-text embedding, POST-projection, L2-normalized (512-d)",
}
DEFAULT_CLIP_FEATURE = "cls"


def load_clip_feature_model(
    ckpt_path,
    *,
    clip_feature: str = DEFAULT_CLIP_FEATURE,
    device="cpu",
    config_preset: str = "vit_b_32",
):
    """Load a CLIP Lightning checkpoint, wrapped so ``forward(image)`` returns
    the representation named by `clip_feature`.

    Goes through `load_clip_lightning`, which requires the checkpoint to match the
    model's state dict exactly, so a wrong-architecture checkpoint fails loudly
    instead of silently producing embeddings from a half-initialized model.

    Returns a NEW module each call (the underlying CLIP is constructed per
    checkpoint), so callers must use the returned reference rather than holding
    onto one from an earlier call.
    """
    if clip_feature not in CLIP_FEATURES:
        raise ValueError(
            f"clip_feature must be one of {sorted(CLIP_FEATURES)}, got {clip_feature!r}."
        )

    from temporal_perturbations.models.clip_yfcc.ckpt_loader import load_clip_lightning
    from temporal_perturbations.models.clip_yfcc.model import (
        CLIPVisionCLSExtractor,
        CLIPVisionProjectedExtractor,
    )

    clip_model = load_clip_lightning(
        ckpt_path=str(ckpt_path),
        config_preset=config_preset,
        map_location=device,
    )
    clip_model.to(device).eval()

    extractor = CLIPVisionCLSExtractor if clip_feature == "cls" else CLIPVisionProjectedExtractor
    return extractor(clip_model).to(device).eval()
