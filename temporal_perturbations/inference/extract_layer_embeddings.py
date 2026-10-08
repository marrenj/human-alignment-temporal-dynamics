"""Per-layer feature extraction for layer-reweighted RSA (FR-RSA).

Mirrors :func:`extract_embeddings.extract_embeddings` (same dataset dispatch,
autocast, loader) but instead of collecting the model's single final embedding it
drives a :class:`frrsa_core.FeatureExtractor` over intermediate layers, collapses
each layer's features to the reference-RDM condition set (concept for THINGS,
synset for NOD, alphabetical to match the ROI RDM row order), and returns one RDM
upper-triangle vector per layer — ready for :func:`frrsa_core.fit_frrsa`.

Condition-averaging is done with **incremental running sums** so the full
flattened conv features (e.g. ResNet ``layer1`` is 802,816-dim) are never all held
in memory at once — only the per-condition means (``n_conditions × D`` per layer)
survive the loop, after which each layer's RDM is computed and the features freed.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from temporal_perturbations.data.things_dataset import ThingsFMRIDataset
from temporal_perturbations.data.nod_dataset import NodDataset
from temporal_perturbations.inference.frrsa_core import (
    FeatureExtractor,
    compute_layer_rdm_vector,
    select_layers,
)


def _build_neural_dataset(dataset_name, img_dir, annotations_file, max_images_per_category,
                          transform=None):
    """Build the condition-labelled dataset used for neural RSA (THINGS-fMRI or NOD).

    `transform=None` leaves the dataset classes' historical THINGS-statistics
    squash-resize in place. FR-RSA runs on ImageNet-supervised and CLIP-contrastive
    checkpoints, none of which were trained under that, so callers should pass the
    recipe from `data.eval_transforms`.
    """
    if dataset_name == "things":
        cols = pd.read_csv(annotations_file, nrows=1).columns
        if "concept" not in cols:
            raise ValueError(
                "FR-RSA on THINGS requires the fMRI annotations CSV (with a 'concept' "
                "column); got a behavioral CSV without it."
            )
        return ThingsFMRIDataset(img_annotations_file=annotations_file, img_dir=img_dir,
                                 transform=transform)
    if dataset_name == "nod":
        return NodDataset(
            category_index_file=annotations_file,
            img_dir=img_dir,
            max_images_per_category=max_images_per_category,
            transform=transform,
        )
    raise ValueError(f"FR-RSA supports dataset 'things' or 'nod'; got {dataset_name!r}.")


def build_neural_stimulus_cache(dataset_name, img_dir, annotations_file,
                                max_images_per_category, preprocessing,
                                batch_size, num_workers, logger=None):
    """Decode a neural stimulus set ONCE into a list of (images, categories) batches.

    Exists so a sweep can evaluate many checkpoints without re-decoding. THINGS-fMRI
    is 8,640 JPEGs at ~14 img/s -- 10.2 minutes -- and FR-RSA used
    to pay that on every (run, epoch, dataset) job. The batches are exactly what the
    DataLoader would have yielded, so `extract_layer_rdm_vectors` consumes either
    interchangeably.

    Held as CPU float32: ~5.2 GB for THINGS-fMRI, ~0.6 GB for NOD. Build one per
    (dataset, preprocessing recipe) -- the recipes differ per architecture, so a
    cache is only reusable across checkpoints that share one.
    """
    dataset = _build_neural_dataset(
        dataset_name, img_dir, annotations_file, max_images_per_category,
        transform=preprocessing.build(224) if preprocessing is not None else None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    batches, n = [], 0
    for batch in tqdm(loader, desc=f"cache {dataset_name}", total=len(loader)):
        if len(batch) != 3:
            raise ValueError(
                "FR-RSA neural extraction expects (names, images, categories) batches."
            )
        _names, images, categories = batch
        batches.append((images, list(categories)))
        n += images.shape[0]
    if logger:
        gb = sum(b[0].element_size() * b[0].nelement() for b in batches) / 1e9
        logger.info("  [cache] %s: %d stimuli in %d batches (%.2f GB, preprocessing=%s)",
                    dataset_name, n, len(batches), gb,
                    preprocessing.tag if preprocessing else "dataset default")
    return batches


def extract_layer_rdm_vectors(
    model,
    dataset_name: str,
    img_dir: str,
    annotations_file: str,
    batch_size: int,
    num_workers: int,
    max_images_per_category: Optional[int],
    device: torch.device,
    architecture: Optional[str] = None,
    preprocessing=None,
    layer_rdm_metric: str = "cosine",
    rdm_device: Optional[torch.device] = None,
    stimulus_cache=None,
    logger=None,
) -> Dict[str, object]:
    """Run the dataset through ``model`` and return one RDM vector per layer.

    ``preprocessing`` is an ``EvalPreprocessing`` (see ``data.eval_transforms``) giving
    the eval transform this checkpoint was trained under. ``None`` falls back to the
    dataset classes' historical THINGS-statistics squash-resize, which matches no
    training recipe for the architectures FR-RSA covers -- pass it explicitly.

    Returns dict with:
        layer_rdm_vectors -- {layer_label -> upper-tri RDM vector (len n_cond*(n_cond-1)/2)}
        condition_labels  -- alphabetical concept/synset list (row order of every RDM)
        n_conditions      -- number of conditions
        layer_labels      -- extraction layers, in model order
    """
    pin_memory = device.type == "cuda"
    if stimulus_cache is not None:
        # Pre-decoded by build_neural_stimulus_cache; identical batches, no image IO.
        loader = stimulus_cache
    else:
        if img_dir is None:
            raise ValueError("img_dir must be provided for FR-RSA feature extraction.")
        dataset = _build_neural_dataset(
            dataset_name, img_dir, annotations_file, max_images_per_category,
            transform=preprocessing.build(224) if preprocessing is not None else None,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    model = model.to(device)
    model.eval()

    layer_defs = select_layers(model, architecture=architecture)
    layer_labels = [label for label, _ in layer_defs]
    if logger:
        logger.info("FR-RSA layers (%d): %s", len(layer_labels), ", ".join(layer_labels))

    use_autocast = device.type == "cuda"
    autocast_ctx = torch.amp.autocast if use_autocast else nullcontext

    # Incremental per-condition running sums: sums[label][category] -> float32 vector;
    # counts[category] -> int (shared across layers, same samples). float32 keeps the
    # block-level footprint bounded (ConvNeXt-B ~36 flattened-conv blocks); summing
    # <=12 exemplars per THINGS concept in float32 is numerically negligible.
    sums: Dict[str, Dict[str, np.ndarray]] = {label: {} for label in layer_labels}
    counts: Dict[str, int] = {}

    n_seen = 0
    with FeatureExtractor(layer_defs) as extractor, torch.no_grad():
        progress = tqdm(loader, desc=f"FR-RSA extract {dataset_name}", total=len(loader))
        for batch in progress:
            if len(batch) == 2:
                images, categories = batch          # pre-decoded stimulus cache
            elif len(batch) == 3:
                _names, images, categories = batch  # live DataLoader
            else:
                raise ValueError(
                    "FR-RSA neural extraction expects (names, images, categories) "
                    "batches from a DataLoader, or (images, categories) from a cache."
                )
            categories = list(categories)
            images = images.to(device, non_blocking=pin_memory)

            with autocast_ctx(device_type="cuda") if use_autocast else autocast_ctx():
                model(images)
            feats = extractor.pop()  # {label -> [B, D] cpu float32}
            n_seen += len(categories)

            # Accumulate per-category sums, once per unique category in the batch.
            uniq = set(categories)
            cat_idx = {c: [i for i, cc in enumerate(categories) if cc == c] for c in uniq}
            for c in uniq:
                counts[c] = counts.get(c, 0) + len(cat_idx[c])
            for label in layer_labels:
                fb = feats[label].numpy()
                store = sums[label]
                for c in uniq:
                    s = fb[cat_idx[c]].sum(axis=0, dtype=np.float32)
                    store[c] = store[c] + s if c in store else s

    condition_labels = sorted(counts.keys())
    n_conditions = len(condition_labels)
    if logger:
        logger.info("FR-RSA: %d stimuli -> %d conditions", n_seen, n_conditions)

    # Per layer: assemble [n_conditions, D] condition-mean matrix, compute its RDM
    # vector, then free the matrix before moving to the next layer.
    layer_rdm_vectors: Dict[str, np.ndarray] = {}
    for label in layer_labels:
        store = sums[label]
        matrix = np.stack([store[c] / counts[c] for c in condition_labels]).astype(np.float32)
        layer_rdm_vectors[label] = compute_layer_rdm_vector(
            matrix, metric=layer_rdm_metric, device=rdm_device
        )
        sums[label] = {}  # free
        del matrix

    return {
        "layer_rdm_vectors": layer_rdm_vectors,
        "condition_labels": condition_labels,
        "n_conditions": n_conditions,
        "layer_labels": layer_labels,
    }
