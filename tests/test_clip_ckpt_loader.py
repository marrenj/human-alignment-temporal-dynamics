"""Pin that CLIP checkpoint loads use strict=True (rejects missing, unexpected, and mismatched keys)."""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(4, 4)
        self.text = nn.Linear(4, 4)
        self.logit_scale = nn.Parameter(torch.zeros(()))


def _same(model, state_dict):
    return all(torch.equal(model.state_dict()[k], v) for k, v in state_dict.items())


def test_strict_true_loads_every_tensor():
    src, dst = TinyCLIP(), TinyCLIP()
    dst.load_state_dict(src.state_dict(), strict=True)
    assert _same(dst, src.state_dict())


def test_strict_true_rejects_missing_key():
    sd = TinyCLIP().state_dict()
    del sd["logit_scale"]
    with pytest.raises(RuntimeError, match="Missing key"):
        TinyCLIP().load_state_dict(sd, strict=True)


def test_strict_true_rejects_unexpected_key():
    sd = TinyCLIP().state_dict()
    sd["visual.proj"] = torch.zeros(4, 4)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        TinyCLIP().load_state_dict(sd, strict=True)


def test_strict_true_rejects_shape_mismatch():
    sd = TinyCLIP().state_dict()
    sd["visual.weight"] = torch.zeros(5, 4)
    with pytest.raises(RuntimeError, match="size mismatch"):
        TinyCLIP().load_state_dict(sd, strict=True)
