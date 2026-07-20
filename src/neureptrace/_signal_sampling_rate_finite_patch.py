"""Guard time-axis helpers against non-finite derived sampling rates."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_signal_sampling_rate_finite_patch_installed"


def install() -> None:
    """Patch sampling-rate helpers to reject reciprocal overflow."""

    band = importlib.import_module("neureptrace.signal.band")
    if getattr(band, _PATCH_MARKER, False):
        return

    original = band.sampling_rate_from_time_axis

    @wraps(original)
    def sampling_rate_from_time_axis(time_vector: Any) -> float:
        sampling_rate = float(original(time_vector))
        if not np.isfinite(sampling_rate):
            raise ValueError("time_vector sample interval must yield a finite sampling rate.")
        return sampling_rate

    band.sampling_rate_from_time_axis = sampling_rate_from_time_axis
    band.sampling_rate_from_time_vector = sampling_rate_from_time_axis
    setattr(band, _PATCH_MARKER, True)


__all__ = ["install"]
