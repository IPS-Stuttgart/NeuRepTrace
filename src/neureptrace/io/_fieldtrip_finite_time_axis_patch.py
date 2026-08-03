"""Reject non-finite FieldTrip time coordinates before MNE construction."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_fieldtrip_finite_time_axis_patch_installed"


def _finite_time_vectors(times: Any) -> None:
    for index, time in enumerate(times):
        values = np.asarray(time, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Time vector {index} must contain only finite values.")


def install() -> None:
    """Validate normalized FieldTrip time vectors before shape comparisons."""

    fieldtrip_mat = importlib.import_module("neureptrace.io.fieldtrip_mat")
    original = fieldtrip_mat._validate_trials_and_times
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _validate_trials_and_times(trials, times, *, require_equal_lengths: bool) -> None:
        _finite_time_vectors(times)
        original(trials, times, require_equal_lengths=require_equal_lengths)

    setattr(_validate_trials_and_times, _PATCH_MARKER, True)
    _validate_trials_and_times.__wrapped__ = original
    fieldtrip_mat._validate_trials_and_times = _validate_trials_and_times


__all__ = ["install"]
