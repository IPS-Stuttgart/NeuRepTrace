"""Reject textual and binary values in phase-aggregation helpers."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

_TEXT_OR_BINARY_TYPES = (str, bytes, bytearray, memoryview)
_PATCH_MARKER = "_neureptrace_signal_phase_text_validation_patch_installed"


def install() -> None:
    """Patch phase helpers so numeric-looking text is not treated as phase data."""

    band = importlib.import_module("neureptrace.signal.band")
    if getattr(band, _PATCH_MARKER, False):
        return

    original_circular_mean_phase = band.circular_mean_phase
    original_average_phases = band.average_phases

    @wraps(original_circular_mean_phase)
    def circular_mean_phase(phases: Any, *, axis: Any = None):
        if isinstance(phases, _TEXT_OR_BINARY_TYPES):
            raise ValueError(
                "phases must contain numeric phase values, not text or binary data."
            )
        return original_circular_mean_phase(phases, axis=axis)

    @wraps(original_average_phases)
    def average_phases(phases: Any):
        if isinstance(phases, _TEXT_OR_BINARY_TYPES):
            raise ValueError(
                "phases must be a collection of numeric phase arrays, not text or binary data."
            )
        return original_average_phases(phases)

    band.circular_mean_phase = circular_mean_phase
    band.average_phases = average_phases
    setattr(band, _PATCH_MARKER, True)


__all__ = ["install"]
