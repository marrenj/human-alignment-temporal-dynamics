"""Cross-architecture model loading and feature extraction for the `runs_root`
checkpoint families (RN50, ViT-B/16, ConvNeXt-B, CLIP-ViT-B/32).
"""
from __future__ import annotations

import gc
import pickle
import re
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn

from temporal_perturbations.models.factory import _TIMM_ARCH_MAP

# Run-directory prefix -> the architecture key understood by _TIMM_ARCH_MAP.
# The "imagenet" in these prefixes is the literal on-disk run-dir name (what the
# models were trained on); the eval itself uses the 48 THINGS images.
ARCH_FROM_RUNDIR = {
    "resnet50_imagenet":   "RN50",
    "convnext_b_imagenet": "ConvNeXt-B",
    "vitb16_imagenet":     "ViT-B/16",
}

_EPOCH_RE = re.compile(r"epoch_(?:epoch=)?(\d+)\.ckpt$")
_SUBEPOCH_RE = re.compile(r"epoch_(\d+)_step_(\d+)\.ckpt$")


def architecture_for_run_dir(run_dir: Path) -> str:
    """Infer the architecture key from a run directory name like
    `resnet50_imagenet_label_shuffle_early`."""
    name = Path(run_dir).name
    for prefix, arch in ARCH_FROM_RUNDIR.items():
        if name.startswith(prefix):
            return arch
    raise ValueError(
        f"Cannot infer architecture from run dir '{name}'. "
        f"Expected one of prefixes: {sorted(ARCH_FROM_RUNDIR)}"
    )


def parse_epoch_from_scratch_filename(name: str) -> int:
    """Epoch integer from either `epoch_NNN.ckpt` or `epoch_epoch=NNN.ckpt`."""
    m = _EPOCH_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse epoch from checkpoint filename: {name}")
    return int(m.group(1))


def sorted_scratch_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(epoch, path), ...] sorted by epoch, one entry per epoch.

    Some baseline runs were resumed and saved the same epoch under two names
    (`epoch_009.ckpt` and `epoch_epoch=009.ckpt` — identical weights). When both
    exist we keep the clean `epoch_NNN.ckpt` form deterministically.
    """
    ckpt_dir = Path(ckpt_dir)
    best: dict[int, Path] = {}
    for p in ckpt_dir.glob("epoch_*.ckpt"):
        epoch = parse_epoch_from_scratch_filename(p.name)
        incumbent = best.get(epoch)
        if incumbent is None:
            best[epoch] = p
            continue
        # Prefer the clean `epoch_NNN.ckpt` over `epoch_epoch=NNN.ckpt`.
        if "epoch=" in incumbent.name and "epoch=" not in p.name:
            best[epoch] = p
    return sorted(best.items())


def parse_subepoch_from_filename(name: str) -> tuple[int, int]:
    """(epoch, step) from ``epoch_006_step_00010.ckpt``."""
    m = _SUBEPOCH_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse sub-epoch checkpoint filename: {name}")
    return int(m.group(1)), int(m.group(2))


def sorted_subepoch_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] sorted by step for a sub_epoch directory."""
    ckpt_dir = Path(ckpt_dir)
    results: dict[int, Path] = {}
    for p in ckpt_dir.glob("epoch_*_step_*.ckpt"):
        _epoch, step = parse_subepoch_from_filename(p.name)
        if step not in results or str(p) < str(results[step]):
            results[step] = p
    return sorted(results.items())


def subepoch_parent_epoch(ckpt_dir: Path) -> int:
    """The single parent epoch shared by all sub-epoch checkpoints in *ckpt_dir*."""
    epochs: set[int] = set()
    for p in Path(ckpt_dir).glob("epoch_*_step_*.ckpt"):
        epoch, _ = parse_subepoch_from_filename(p.name)
        epochs.add(epoch)
    if not epochs:
        raise FileNotFoundError(f"No sub-epoch checkpoints in {ckpt_dir}")
    if len(epochs) > 1:
        raise ValueError(
            f"Sub-epoch checkpoints span multiple parent epochs {sorted(epochs)} in {ckpt_dir}"
        )
    return epochs.pop()


def available_scratch_epochs(run_dir: Path) -> set[int]:
    """Every epoch with a checkpoint under `<run_dir>/checkpoints/all_epochs`."""
    return {e for e, _ in sorted_scratch_checkpoints(Path(run_dir) / "checkpoints" / "all_epochs")}


def scratch_checkpoint_at_epoch(run_dir: Path, epoch: int) -> Path:
    """Resolve one epoch's checkpoint under `<run_dir>/checkpoints/all_epochs`."""
    ckpts = dict(sorted_scratch_checkpoints(Path(run_dir) / "checkpoints" / "all_epochs"))
    if epoch not in ckpts:
        raise FileNotFoundError(f"No checkpoint for epoch {epoch} under {run_dir}")
    return ckpts[epoch]


class _Stub:
    """Placeholder for any pickled global that cannot be imported (the
    Lightning checkpoint embeds a `config.supervised_config.*` object)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


class _TolerantPickleModule:
    Unpickler = _TolerantUnpickler

    @staticmethod
    def load(f, **kwargs):
        return _TolerantUnpickler(f).load()


def load_timm_state_dict_from_lightning(ckpt_path: str, map_location="cpu") -> dict:
    """Extract the timm backbone state_dict (with the `model.` prefix stripped)
    from a from-scratch Lightning checkpoint.

    Reads the whole file in one bulk sequential pass (mmap=False). On network
    filesystems mmap's page-fault-driven reads are ~2-4x
    slower than a bulk read even though they touch fewer bytes — sequential
    throughput dominates. The extra optimizer/loop blobs are read but discarded;
    the returned tensors are detached CPU clones so peak RAM stays at ~one
    checkpoint's worth per epoch.
    """
    ckpt = torch.load(
        ckpt_path,
        map_location=map_location,
        weights_only=False,
        mmap=False,
        pickle_module=_TolerantPickleModule,
    )
    if "state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint has no 'state_dict' key "
            f"(top-level keys: {sorted(ckpt.keys())}): {ckpt_path}"
        )
    state_dict = ckpt["state_dict"]
    return {
        (k[len("model."):] if k.startswith("model.") else k): v.detach().clone()
        for k, v in state_dict.items()
        if k.startswith("model.")
    }


class TimmFeatureWrapper(nn.Module):
    """Wrap a timm classifier so the forward pass returns the pooled penultimate
    (pre-logits) feature rather than class logits."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.forward_head(self.net.forward_features(x), pre_logits=True)


def build_timm_feature_model(architecture: str, device, num_classes: int = 1000) -> TimmFeatureWrapper:
    """Build the timm backbone for `architecture` (an _TIMM_ARCH_MAP key) and
    wrap it for penultimate-feature extraction. Weights are loaded per-epoch by
    the caller via `model.net.load_state_dict`."""
    if architecture not in _TIMM_ARCH_MAP:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Expected one of {sorted(_TIMM_ARCH_MAP)}"
        )
    net = timm.create_model(_TIMM_ARCH_MAP[architecture], pretrained=False, num_classes=num_classes)
    model = TimmFeatureWrapper(net).to(device)
    model.eval()
    return model


CLIP_ARCH = "CLIP ViT-B/32"

# Architecture key -> run-directory prefix for the `runs_root` families.
ARCH_PREFIX = {
    "RN50":          "resnet50_imagenet",
    "ViT-B/16":      "vitb16_imagenet",
    "ConvNeXt-B":    "convnext_b_imagenet",
    CLIP_ARCH:       "clip_vitb32",
}


def build_cross_arch_model(
    architecture: str,
    device,
    clip_feature: str = "cls",
) -> Tuple[Optional[nn.Module], Callable[[str], nn.Module]]:
    """Build a model whose ``forward(images)`` returns the penultimate feature,
    plus a ``load_weights(ckpt_path) -> model`` closure for per-epoch weight
    swaps.

    For the three timm architectures the closure mutates and returns the same
    object. For CLIP it builds a fresh model per checkpoint, so callers must
    use the returned reference.
    """
    if architecture == CLIP_ARCH:
        from temporal_perturbations.models.clip_yfcc import load_clip_feature_model

        def load_weights(ckpt_path):
            model = load_clip_feature_model(
                ckpt_path, clip_feature=clip_feature, device=device,
            )
            gc.collect()
            return model

        return None, load_weights

    model = build_timm_feature_model(architecture, device)

    def load_weights(ckpt_path):
        sd = load_timm_state_dict_from_lightning(str(ckpt_path), map_location="cpu")
        model.net.load_state_dict(sd, strict=True)
        model.eval()
        del sd
        gc.collect()
        return model

    return model, load_weights


def embed_batched(
    model: nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device,
) -> np.ndarray:
    """Forward ``images`` through ``model`` in batches, returning a numpy array.

    Uses mixed-precision autocast on CUDA.
    """
    use_autocast = device.type == "cuda"
    out = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i:i + batch_size].to(device, non_blocking=False)
            if use_autocast:
                with torch.amp.autocast("cuda"):
                    feats = model(chunk)
            else:
                feats = model(chunk)
            out.append(feats.float().cpu())
    return torch.cat(out, dim=0).numpy()
