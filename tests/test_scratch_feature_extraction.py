"""Feature extraction must go through TimmFeatureWrapper, never a raw timm model.

`_extract_features` raises TypeError if the model exposes `forward_features` —
hand-rolled pooling skips ConvNeXt-B's post-pool LayerNorm, producing silently
wrong 1024-d features. `TimmFeatureWrapper` calls timm's own
`forward_head(..., pre_logits=True)`, which is correct for all architectures.
"""

import logging

import numpy as np
import pytest
import timm
import torch
import torch.nn as nn

from temporal_perturbations.inference.linear_probe_eval import _extract_features
from temporal_perturbations.models.factory import _TIMM_ARCH_MAP
from temporal_perturbations.models.scratch.scratch_utils import TimmFeatureWrapper


LOG = logging.getLogger("test_scratch_feature_extraction")
ARCHITECTURES = ["RN50", "ViT-B/16", "ConvNeXt-B"]


def _one_batch_loader(n=2, size=224, seed=0):
    """`_extract_features` only needs an iterable of (images, labels) with a len()."""
    torch.manual_seed(seed)
    return [(torch.randn(n, 3, size, size), torch.zeros(n, dtype=torch.long))]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_wrapped_matches_timm_pre_logits(architecture):
    """The wrapper is the reference: it is literally timm's pre_logits head."""
    net = timm.create_model(_TIMM_ARCH_MAP[architecture], pretrained=False, num_classes=1000)
    net.eval()
    loader = _one_batch_loader()

    wrapped, _ = _extract_features(TimmFeatureWrapper(net), loader, torch.device("cpu"), LOG)
    with torch.no_grad():
        direct = net.forward_head(net.forward_features(loader[0][0]), pre_logits=True).numpy()

    assert wrapped.shape == direct.shape
    np.testing.assert_allclose(wrapped, direct, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_raw_timm_model_raises(architecture):
    """A raw timm model exposes forward_features, so _extract_features must reject it."""
    net = timm.create_model(_TIMM_ARCH_MAP[architecture], pretrained=False, num_classes=1000)
    net.eval()
    loader = _one_batch_loader()

    with pytest.raises(TypeError, match="Wrap it in TimmFeatureWrapper"):
        _extract_features(net, loader, torch.device("cpu"), LOG)


def test_dataparallel_silently_yields_logits_not_features():
    """A model whose forward() returns logits (not features) passes the guard but
    produces 1000-d class logits where the probe expects 2048-d features — a silent
    shape-compatible error. TimmFeatureWrapper is the fix; DataParallel(raw) is the
    motivating example (it hides forward_features from hasattr but delegates forward
    to the raw model's full pipeline including the classifier head).
    """
    net = timm.create_model(_TIMM_ARCH_MAP["RN50"], pretrained=False, num_classes=1000)
    net.eval()
    parallel = nn.DataParallel(net)
    assert not hasattr(parallel, "forward_features")

    # DataParallel itself cannot run here -- with CUDA present it demands its module be
    # on cuda:0 -- so the leak is exercised through a stand-in with the same shape of
    # interface: a wrapper that forwards to the net but does not expose forward_features.
    class _NoForwardFeatures(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x):
            return self.module(x)

    leaked, _ = _extract_features(_NoForwardFeatures(net), _one_batch_loader(),
                                  torch.device("cpu"), LOG)
    features, _ = _extract_features(TimmFeatureWrapper(net), _one_batch_loader(),
                                    torch.device("cpu"), LOG)
    assert leaked.shape[1] == 1000, "expected the silent logits leak this guards against"
    assert features.shape[1] == 2048

    with pytest.raises(AttributeError):
        TimmFeatureWrapper(parallel).eval()(torch.randn(1, 3, 224, 224))
