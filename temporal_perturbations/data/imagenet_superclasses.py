"""16-superordinate-category mapping for ImageNet-1k classes.

Assigns each of the 1000 ImageNet synsets to one of 16 coarse categories
(15 named + "Uncategorized") using WordNet hypernym paths.  Priority follows
dict insertion order: e.g. fruits go to "Food" (not "Plant Part"), golfcarts
go to "Vehicle" (not "Sports Equipment").

The mapping is computed via NLTK WordNet on first call and cached to a JSON
file so subsequent calls skip the hypernym walk entirely.

Shared by the TriMap scripts (trimap_imagenet.py, fig3_storyboard.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CATEGORY_WORDS = {
    "Animal":               "animal",
    "Vehicle":              "vehicle",
    "Clothing":             "clothing",
    "Food":                 "food",
    "Musical Instrument":   "musical_instrument",
    "Plant Part":           "plant_part",
    "Structure":            "structure",
    "Tool":                 "tool",
    "Furnishing":           "furnishing",
    "Sports Equipment":     "sports_equipment",
    "Appliance":            "appliance",
    "Geological Formation": "geological_formation",
    "Utensil":              "utensil",
    "Electronic Equipment": "electronic_equipment",
    "Measuring Instrument": "measuring_instrument",
}

CATEGORY_NAMES = list(CATEGORY_WORDS.keys()) + ["Uncategorized"]

CATEGORY_PALETTE = {
    "Animal":               "#1f77b4",
    "Vehicle":              "#e377c2",
    "Clothing":             "#ff7f0e",
    "Food":                 "#2ca02c",
    "Musical Instrument":   "#9467bd",
    "Plant Part":           "#17becf",
    "Structure":            "#bcbd22",
    "Tool":                 "#7f7f7f",
    "Furnishing":           "#d62728",
    "Sports Equipment":     "#8c564b",
    "Appliance":            "#aec7e8",
    "Geological Formation": "#ffbb78",
    "Utensil":              "#98df8a",
    "Electronic Equipment": "#ff9896",
    "Measuring Instrument": "#c5b0d5",
    "Uncategorized":        "#bfbfbf",
}

_CACHE_PATH = Path(__file__).resolve().parent / "_imagenet_superclass_cache.json"

_synset_to_category: dict[str, str] | None = None
_class_idx_to_superclass_idx: dict[int, int] | None = None


def _build_synset_map() -> dict[str, str]:
    from nltk.corpus import wordnet as wn

    cat_synsets = {
        label: set(wn.synsets(word, pos="n"))
        for label, word in CATEGORY_WORDS.items()
    }

    all_synsets = list(wn.all_synsets(pos="n"))
    imagenet_synsets = [
        s for s in all_synsets
        if hasattr(s, "offset") or hasattr(s, "_offset")
    ]

    mapping: dict[str, str] = {}
    for syn in wn.all_synsets(pos="n"):
        offset = syn.offset()
        sid = f"n{offset:08d}"
        ancestors = set()
        for path in syn.hypernym_paths():
            ancestors.update(path)
        ancestors.add(syn)
        cat = "Uncategorized"
        for label, syns in cat_synsets.items():
            if syns & ancestors:
                cat = label
                break
        mapping[sid] = cat

    return mapping


def load_synset_to_category() -> dict[str, str]:
    global _synset_to_category
    if _synset_to_category is not None:
        return _synset_to_category

    if _CACHE_PATH.exists():
        with _CACHE_PATH.open() as f:
            _synset_to_category = json.load(f)
        logger.info("Loaded superclass cache from %s (%d synsets)",
                     _CACHE_PATH, len(_synset_to_category))
        return _synset_to_category

    logger.info("Building ImageNet superclass mapping via WordNet (one-time)...")
    _synset_to_category = _build_synset_map()
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w") as f:
        json.dump(_synset_to_category, f)
    logger.info("Cached superclass mapping to %s (%d synsets)",
                 _CACHE_PATH, len(_synset_to_category))
    return _synset_to_category


def build_class_idx_to_superclass(class_to_idx: dict[str, int]) -> dict[int, int]:
    """Map ImageNet class indices to superclass indices (0..15).

    `class_to_idx` is the mapping from synset-ID folder names (e.g. 'n02119789')
    to integer class labels, as produced by torchvision's ImageFolder or
    clip_benchmark's imagenet1k builder.
    """
    synset_map = load_synset_to_category()
    cat_to_idx = {c: i for i, c in enumerate(CATEGORY_NAMES)}
    result = {}
    for synset_id, class_idx in class_to_idx.items():
        cat = synset_map.get(synset_id, "Uncategorized")
        result[class_idx] = cat_to_idx[cat]
    return result


def make_superclass_target_transform(class_to_idx: dict[str, int]):
    """Return a callable that maps an original class label to a superclass label."""
    mapping = build_class_idx_to_superclass(class_to_idx)
    def transform(label):
        return mapping[label]
    return transform
