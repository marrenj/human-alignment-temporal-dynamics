"""Robust loader for the from-scratch PyTorch-Lightning checkpoints.

The cross-architecture checkpoints under the ``runs_root`` tree were produced by
`train_from_scratch.py` (its LightningModule wraps
the network as ``self.model = ...``, and pickles config dataclasses from a
``config.*`` package alongside the weights). Two problems for a plain
``torch.load``:

  1. ``ModuleNotFoundError: No module named 'config'`` — the pickled
     ``hyper_parameters`` reference the trainer's top-level ``config`` package, which
     isn't importable under that name here.
  2. The weights are nested under ``state_dict`` with a ``model.`` prefix (the
     LightningModule attribute the model lives on).

This module solves both. A permissive unpickler maps ANY unresolvable class to a
throwaway placeholder — we only want the tensors, never the config objects — and
the ``model.`` prefix is stripped. It needs no knowledge of the pickled config
classes, so the same loader serves all four from-scratch families.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Dict

import torch


class _Placeholder:
    """Stand-in for any class we can't (and don't need to) import while unpickling."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _PermissiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Placeholder


class _PermissivePickleModule:
    """pickle-module shim for ``torch.load(pickle_module=...)`` that never fails on
    an unknown class (returns :class:`_Placeholder` instead)."""

    Unpickler = _PermissiveUnpickler
    Pickler = pickle.Pickler
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL

    @staticmethod
    def load(f, **_kw):
        return _PermissiveUnpickler(f).load()

    @staticmethod
    def loads(data, **_kw):
        return _PermissiveUnpickler(io.BytesIO(data)).load()

    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def load_lightning_state_dict(
    ckpt_path: str | Path,
    map_location="cpu",
    strip_prefix: str = "model.",
) -> Dict[str, "torch.Tensor"]:
    """Return the inner model ``state_dict`` from a Lightning ``.ckpt``."""
    obj = torch.load(
        str(ckpt_path),
        map_location=map_location,
        pickle_module=_PermissivePickleModule,
        weights_only=False,
    )
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Expected a dict from torch.load, got {type(obj).__name__}: {ckpt_path}"
        )
    if "state_dict" not in obj:
        raise KeyError(
            f"Checkpoint {ckpt_path} has no 'state_dict' key "
            f"(top-level keys: {sorted(obj.keys())})."
        )
    state = obj["state_dict"]
    inner = {k[len(strip_prefix):]: v for k, v in state.items() if k.startswith(strip_prefix)}
    if not inner:
        raise ValueError(
            f"No keys in state_dict start with {strip_prefix!r} "
            f"(first 5 keys: {sorted(state.keys())[:5]}). "
            f"Checkpoint: {ckpt_path}"
        )
    return inner
