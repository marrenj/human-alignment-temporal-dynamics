"""Pin the DoRA loader's coverage check: an adapter checkpoint must match the model's adapters exactly."""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_perturbations.models.clip_hba.clip_hba_utils import (  # noqa: E402
    DoRALayer,
    dora_parameter_keys,
    load_dora_checkpoint,
    load_dora_state_dict,
)


class ThreeBlockModel(nn.Module):
    """Three linear blocks; the last ``n_adapted`` are wrapped in DoRA, mirroring apply_dora_to_ViT."""

    def __init__(self, n_adapted=2, r=4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        for i in range(3 - n_adapted, 3):
            self.blocks[i] = DoRALayer(self.blocks[i], r=r, dora_dropout=0.0)


def _adapter_state(model):
    return {k: v.clone() for k, v in model.state_dict().items() if k in dora_parameter_keys(model)}


def test_expected_keys_enumerate_only_adapted_blocks():
    assert dora_parameter_keys(ThreeBlockModel()) == {
        f"blocks.{i}.{p}" for i in (1, 2) for p in ("m", "delta_D_A", "delta_D_B")
    }


def test_matching_checkpoint_loads_every_adapter_tensor():
    src, dst = ThreeBlockModel(), ThreeBlockModel()
    load_dora_state_dict(dst, _adapter_state(src), source="test")
    for key, value in _adapter_state(src).items():
        assert torch.equal(dst.state_dict()[key], value)


def test_extra_key_from_more_adapted_layers_raises():
    with pytest.raises(RuntimeError, match=r"unexpected keys: \['blocks\.0"):
        load_dora_state_dict(ThreeBlockModel(n_adapted=2), _adapter_state(ThreeBlockModel(n_adapted=3)), source="test")


def test_missing_adapter_key_raises():
    with pytest.raises(RuntimeError, match=r"missing keys: \['blocks\.1"):
        load_dora_state_dict(ThreeBlockModel(n_adapted=2), _adapter_state(ThreeBlockModel(n_adapted=1)), source="test")


def test_dataparallel_prefixed_checkpoint_is_rejected_not_ignored():
    prefixed = {f"module.{k}": v for k, v in _adapter_state(ThreeBlockModel()).items()}
    with pytest.raises(RuntimeError, match="unexpected keys"):
        load_dora_state_dict(ThreeBlockModel(), prefixed, source="test")


def test_dataparallel_wrapped_model_accepts_unprefixed_checkpoint():
    src = ThreeBlockModel()
    wrapped = nn.DataParallel(ThreeBlockModel())
    load_dora_state_dict(wrapped, _adapter_state(src), source="test")
    assert torch.equal(wrapped.module.blocks[2].m, src.blocks[2].m)


def test_wrong_rank_still_raises_on_shape():
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_dora_state_dict(ThreeBlockModel(r=4), _adapter_state(ThreeBlockModel(r=8)), source="test")


def test_load_dora_checkpoint_round_trip(tmp_path):
    src = ThreeBlockModel()
    (tmp_path / "dora_params").mkdir()
    torch.save(_adapter_state(src), tmp_path / "dora_params" / "epoch7_dora_params.pth")
    dst = ThreeBlockModel()
    assert load_dora_checkpoint(dst, tmp_path, 7) == tmp_path / "dora_params" / "epoch7_dora_params.pth"
    assert torch.equal(dst.blocks[2].delta_D_B, src.blocks[2].delta_D_B)
    with pytest.raises(FileNotFoundError):
        load_dora_checkpoint(dst, tmp_path, 8)
