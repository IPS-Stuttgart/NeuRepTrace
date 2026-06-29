"""Reject malformed values for response-window time controls.

Response-window time arguments are scalar controls. NumPy boolean arrays should
not be coerced to ``0``/``1``, and non-scalar arrays should not be accepted as
scalar times.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_response_window_time_validation_patch_installed"


def _coerce_time_control_scalar(value: Any, *, message: str) -> Any:
    """Return a scalar time-control value or raise the public validation error."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
        return value
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
        return value
    return value


def install() -> None:
    """Patch response-window time control validation."""

    response_window_ensemble = importlib.import_module("neureptrace.response_window_ensemble")
    if getattr(response_window_ensemble, _PATCH_MARKER, False):
        return

    original_normalize_response_times = response_window_ensemble._normalize_response_times
    original_validate_optional_output_time = response_window_ensemble._validate_optional_output_time

    @wraps(original_normalize_response_times)
    def _normalize_response_times(response_times: Sequence[float]) -> tuple[float, ...]:
        message = "Response-window times must be finite."
        try:
            values = tuple(response_times)
        except TypeError as exc:
            raise ValueError(message) from exc
        values = tuple(_coerce_time_control_scalar(value, message=message) for value in values)
        return original_normalize_response_times(values)

    @wraps(original_validate_optional_output_time)
    def _validate_optional_output_time(output_time: float | None) -> float | None:
        if output_time is None:
            return original_validate_optional_output_time(output_time)
        output_time = _coerce_time_control_scalar(output_time, message="output_time must be finite when provided.")
        return original_validate_optional_output_time(output_time)

    response_window_ensemble._normalize_response_times = _normalize_response_times
    response_window_ensemble._validate_optional_output_time = _validate_optional_output_time
    setattr(response_window_ensemble, _PATCH_MARKER, True)


__all__ = ["install"]
