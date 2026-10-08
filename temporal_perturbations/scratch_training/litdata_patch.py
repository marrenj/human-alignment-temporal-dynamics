"""Defensive monkey-patches for litdata's streaming reader.

The chunk-cleanup daemon thread in `litdata.streaming.reader` can die silently
on benign races (e.g. a `.cnt` lock-count file removed between stat and open).
When the cleanup thread exits, the local cache stops being garbage-collected;
once it hits the size cap or fills /tmp, the main reader blocks waiting for
free space and training hangs with no log output and zero GPU activity.

Apply once at process start (see `pipeline/scripts/training/train_from_scratch.py`).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    try:
        from litdata.streaming import reader as ldr
    except ImportError:
        return

    _patch_remaining_locks(ldr)
    _patch_maybe_delete_chunks(ldr)
    _APPLIED = True


def _find_owning_class(module, method_name):
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and method_name in obj.__dict__:
            return obj
    return None


_NO_OP_WARNING = (
    "litdata patch: %s.%s not found on any class in litdata.streaming.reader. "
    "Litdata's internals likely changed between versions — the chunk-cleanup "
    "hang fix is NOT installed for this run. Pin litdata exactly (see "
    "requirements.txt) or update scratch_training/litdata_patch.py for the new API."
)


def _patch_remaining_locks(ldr) -> None:
    cls = _find_owning_class(ldr, "_remaining_locks")
    if cls is None:
        logger.warning(_NO_OP_WARNING, "Reader", "_remaining_locks")
        return
    if getattr(cls, "_remaining_locks_patched", False):
        return

    original = cls._remaining_locks

    def _safe_remaining_locks(self, chunk_filepath):
        try:
            return original(self, chunk_filepath)
        except FileNotFoundError:
            return 0

    cls._remaining_locks = _safe_remaining_locks
    cls._remaining_locks_patched = True
    logger.info("litdata patch: wrapped %s._remaining_locks", cls.__name__)


def _patch_maybe_delete_chunks(ldr) -> None:
    cls = _find_owning_class(ldr, "_maybe_delete_chunks")
    if cls is None:
        logger.warning(_NO_OP_WARNING, "Reader", "_maybe_delete_chunks")
        return
    if getattr(cls, "_maybe_delete_chunks_patched", False):
        return

    original = cls._maybe_delete_chunks

    def _safe_maybe_delete_chunks(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except Exception as e:
            logger.warning(
                "litdata _maybe_delete_chunks raised %s: %s; skipping this cleanup pass",
                type(e).__name__, e,
            )
            return None

    cls._maybe_delete_chunks = _safe_maybe_delete_chunks
    cls._maybe_delete_chunks_patched = True
    logger.info("litdata patch: wrapped %s._maybe_delete_chunks", cls.__name__)
