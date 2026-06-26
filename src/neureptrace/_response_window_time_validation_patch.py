"""Reject boolean values for response-window time controls."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_response_window_time_validation_patch_installed"


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def install() -> None:
    """Patch response-window time control validation."""

    response_window_ensemble = importlib.import_module("neureptrace.response_window_ensemble")
    if getattr(response_window_ensemble, _PATCH_MARKER, False):
        return

    original_normalize_response_times = response_window_ensemble._normalize_response_times
    original_validate_optional_output_time = response_window_ensemble._validate_optional_output_time

    @wraps(original_normalize_response_times)
    def _normalize_response_times(response_times: Sequence[float]) -> tuple[float, ...]:
        values = tuple(response_times)
        if any(_is_bool_scalar(value) for value in values):
            raise ValueError("Response-window times must be finite.")
        return original_normalize_response_times(values)

    @wraps(original_validate_optional_output_time)
    def _validate_optional_output_time(output_time: float | None) -> float | None:
        if _is_bool_scalar(output_time):
            raise ValueError("output_time must be finite when provided.")
        return original_validate_optional_output_time(output_time)

    response_window_ensemble._normalize_response_times = _normalize_response_times
    response_window_ensemble._validate_optional_output_time = _validate_optional_output_time
    setattr(response_window_ensemble, _PATCH_MARKER, True)


__all__ = ["install"]
