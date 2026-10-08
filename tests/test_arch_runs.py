"""pipeline/scripts/visualization/arch_runs.py must agree with the eval drivers about how corrected-preprocessing
results are named on disk.

The figure scripts are deliberately stdlib-only, so `PREPROCESSING_TAG` is a copy of
`eval_transforms.ARCH_EVAL_PREPROCESSING`'s tags rather than an import. If the two drift
a figure would silently fall back to "no data" for that architecture.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "scripts" / "visualization"))

import arch_runs  # noqa: E402
from temporal_perturbations.data.eval_transforms import (  # noqa: E402
    ARCH_EVAL_PREPROCESSING,
    THINGS_LEGACY,
)

# Run-dir prefix -> the architecture label the eval drivers key the recipe on.
LABEL = {
    "resnet50_imagenet": "RN50",
    "vitb16_imagenet": "ViT-B/16",
    "convnext_b_imagenet": "ConvNeXt-B",
    arch_runs.CLIP_PREFIX: "CLIP ViT-B/32",
}


@pytest.mark.parametrize("prefix", sorted(LABEL))
def test_preprocessing_tag_matches_eval_transforms(prefix):
    assert arch_runs.PREPROCESSING_TAG[prefix] == ARCH_EVAL_PREPROCESSING[LABEL[prefix]].tag


def test_tag_map_covers_exactly_the_paper_families():
    assert set(arch_runs.PREPROCESSING_TAG) == set(LABEL)


def test_corrected_is_never_the_legacy_recipe():
    # Legacy results are untagged; a corrected suffix that resolved to it would read
    # a file that does not exist.
    for prefix in LABEL:
        assert THINGS_LEGACY.tag not in arch_runs.rdm1854_corrected_suffix(prefix)


def test_rdm1854_suffix_shape():
    # run_behavioral_rsa.py composes `{_variant_suffix}{_preprocessing_suffix}`:
    # CLIP's cls tag precedes the recipe tag, the timm families carry the recipe alone.
    assert arch_runs.rdm1854_corrected_suffix("vitb16_imagenet") == "_in1k_bicubic"
    assert arch_runs.rdm1854_corrected_suffix("resnet50_imagenet") == "_in1k_bilinear"
    assert arch_runs.rdm1854_corrected_suffix(arch_runs.CLIP_PREFIX) == "_cls_clip"


def test_probe_recipe_matches_linear_probe_eval():
    # The figures accept only records stamped with these two markers; if the probe code moves
    # to a new recipe and this copy does not, every figure would keep accepting the old one.
    from temporal_perturbations.inference import linear_probe_eval as lpe
    assert arch_runs.PROBE_C_SELECTION == lpe.PROBE_C_SELECTION
    assert arch_runs.PROBE_FEATURE_SCALING == lpe.PROBE_FEATURE_SCALING


def test_require_current_probe_refuses_older_recipes():
    current = {"c_selection": arch_runs.PROBE_C_SELECTION,
               "feature_scaling": arch_runs.PROBE_FEATURE_SCALING}
    assert arch_runs.require_current_probe(current, "x") is current
    for old in ({"c_selection": "val_sweep"}, {"c_selection": arch_runs.PROBE_C_SELECTION},
                {"feature_scaling": arch_runs.PROBE_FEATURE_SCALING}, {}):
        with pytest.raises(ValueError, match="older recipe"):
            arch_runs.require_current_probe(old, "x")
