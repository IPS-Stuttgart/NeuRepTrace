"""Normalize the trial axis of numeric FieldTrip time matrices."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_fieldtrip_time_matrix_orientation_patched"


def _numeric_time_matrix(times: Any) -> np.ndarray | None:
    """Return a two-dimensional numeric time matrix when one is available."""

    if not isinstance(times, np.ndarray):
        return None
    try:
        matrix = np.asarray(times, dtype=float)
    except (TypeError, ValueError):
        return None
    return matrix if matrix.ndim == 2 else None


def install() -> None:
    """Interpret either axis of a numeric time matrix as the trial axis."""

    fieldtrip_mat = importlib.import_module("neureptrace.io.fieldtrip_mat")
    if getattr(fieldtrip_mat, _PATCH_MARKER, False):
        return

    original_normalize_times = fieldtrip_mat._normalize_times

    def _normalize_times(times: Any, n_trials: int) -> list[np.ndarray]:
        matrix = _numeric_time_matrix(times)
        if matrix is not None and 1 not in matrix.shape:
            if matrix.shape[0] == n_trials:
                return [np.asarray(row, dtype=float).ravel() for row in matrix]
            if matrix.shape[1] == n_trials:
                return [
                    np.asarray(matrix[:, trial_index], dtype=float).ravel()
                    for trial_index in range(n_trials)
                ]
        return original_normalize_times(times, n_trials)

    _normalize_times.__wrapped__ = original_normalize_times
    fieldtrip_mat._normalize_times = _normalize_times
    setattr(fieldtrip_mat, _PATCH_MARKER, True)


__all__ = ["install"]
