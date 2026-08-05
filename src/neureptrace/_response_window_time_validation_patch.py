"""Validate response-window scalar controls and matched-filter template bounds.

Response-window controls are scalar real values. NumPy boolean and complex
scalars must not be coerced to numeric values, non-scalar arrays must not be
accepted as scalar controls, and malformed inputs should consistently raise the
public validation error. Matched-filter template offsets must also remain inside
the configured template window when its width is not divisible by the sampling
step.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_response_window_time_validation_patch_installed"
_MATCHED_FILTER_PATCH_MARKER = "_neureptrace_matched_filter_template_window_patch_installed"


def _coerce_time_control_scalar(value: Any, *, message: str) -> Any:
    """Return a real scalar control value or raise the public validation error."""
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(message)
    return value


def _install_matched_filter_template_window_patch() -> None:
    """Discard rounded template offsets that exceed the requested stop time."""

    matched_filter = importlib.import_module("neureptrace.matched_filter_detection")
    if getattr(matched_filter, _MATCHED_FILTER_PATCH_MARKER, False):
        return

    original_template_offsets = matched_filter._template_offsets

    @wraps(original_template_offsets)
    def _template_offsets(
        template_window: tuple[float, float],
        template_step: float,
    ) -> np.ndarray:
        offsets = np.asarray(
            original_template_offsets(template_window, template_step),
            dtype=float,
        )
        start, stop = map(float, template_window)
        tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(start), abs(stop))
        return offsets[offsets <= stop + tolerance]

    matched_filter._template_offsets = _template_offsets
    setattr(matched_filter, _MATCHED_FILTER_PATCH_MARKER, True)


def install() -> None:
    """Patch response-window scalar validation and matched-filter offsets."""

    _install_matched_filter_template_window_patch()

    response_window_ensemble = importlib.import_module("neureptrace.response_window_ensemble")
    if getattr(response_window_ensemble, _PATCH_MARKER, False):
        return

    original_normalize_response_times = response_window_ensemble._normalize_response_times
    original_validate_optional_output_time = response_window_ensemble._validate_optional_output_time
    original_validate_positive_finite_float = response_window_ensemble._validate_positive_finite_float
    original_validate_positive_integer = response_window_ensemble._validate_positive_integer

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

    @wraps(original_validate_positive_finite_float)
    def _validate_positive_finite_float(value: float, *, name: str) -> float:
        value = _coerce_time_control_scalar(value, message=f"{name} must be positive and finite.")
        return original_validate_positive_finite_float(value, name=name)

    @wraps(original_validate_positive_integer)
    def _validate_positive_integer(value: int, *, name: str) -> int:
        value = _coerce_time_control_scalar(value, message=f"{name} must be a positive integer.")
        return original_validate_positive_integer(value, name=name)

    response_window_ensemble._normalize_response_times = _normalize_response_times
    response_window_ensemble._validate_optional_output_time = _validate_optional_output_time
    response_window_ensemble._validate_positive_finite_float = _validate_positive_finite_float
    response_window_ensemble._validate_positive_integer = _validate_positive_integer
    setattr(response_window_ensemble, _PATCH_MARKER, True)


__all__ = ["install"]
