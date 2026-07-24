"""Reject complex-valued inputs in real signal-processing APIs."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_signal_complex_validation_patch_installed"


def _contains_complex_like(value: object) -> bool:
    """Return whether ``value`` contains a Python or NumPy complex scalar."""

    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return True
        if value.dtype == object:
            return any(_contains_complex_like(item) for item in value.ravel())
        return False
    if hasattr(value, "__array__"):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return False
        if np.issubdtype(array.dtype, np.complexfloating):
            return True
        if array.dtype == object:
            return any(_contains_complex_like(item) for item in array.ravel())
        return False
    if isinstance(value, Sequence):
        return any(_contains_complex_like(item) for item in value)
    return False


def install() -> None:
    """Patch real-valued signal helpers to reject complex inputs before coercion."""

    band = importlib.import_module("neureptrace.signal.band")
    if getattr(band, _PATCH_MARKER, False):
        return

    original_validate_time_axis = band.validate_time_axis
    original_validate_sampling_rate = band.validate_sampling_rate
    original_validate_signal_values = band.validate_signal_values
    original_validate_band_hz = band.validate_band_hz
    original_circular_mean_phase = band.circular_mean_phase
    original_average_phases = band.average_phases

    @wraps(original_validate_time_axis)
    def validate_time_axis(time_vector: Any) -> np.ndarray:
        time_vector = band._materialize_numeric_input(time_vector)
        if _contains_complex_like(time_vector):
            raise ValueError(
                "time_vector must contain real-valued time values, not complex values."
            )
        return original_validate_time_axis(time_vector)

    @wraps(original_validate_sampling_rate)
    def validate_sampling_rate(sampling_rate: Any) -> float:
        sampling_rate = band._materialize_numeric_input(sampling_rate)
        if _contains_complex_like(sampling_rate):
            raise ValueError(
                "sampling_rate must be a real positive finite value, not complex."
            )
        return original_validate_sampling_rate(sampling_rate)

    @wraps(original_validate_signal_values)
    def validate_signal_values(signal_values: Any, *, axis: int = -1) -> np.ndarray:
        signal_values = band._materialize_numeric_input(signal_values)
        if _contains_complex_like(signal_values):
            raise ValueError(
                "signal_values must contain real-valued samples, not complex values."
            )
        return original_validate_signal_values(signal_values, axis=axis)

    @wraps(original_validate_band_hz)
    def validate_band_hz(band_hz: Any, sampling_rate: Any) -> tuple[float, float]:
        band_hz = band._materialize_numeric_input(band_hz)
        sampling_rate = band._materialize_numeric_input(sampling_rate)
        if _contains_complex_like(band_hz):
            raise ValueError("Cutoff frequencies must be real-valued, not complex.")
        if _contains_complex_like(sampling_rate):
            raise ValueError(
                "sampling_rate must be a real positive finite value, not complex."
            )
        return original_validate_band_hz(band_hz, sampling_rate)

    @wraps(original_circular_mean_phase)
    def circular_mean_phase(phases: Any, *, axis: Any = None) -> np.ndarray:
        phases = band._materialize_numeric_input(phases)
        if _contains_complex_like(phases):
            raise ValueError(
                "phases must contain real-valued phase values, not complex values."
            )
        return original_circular_mean_phase(phases, axis=axis)

    @wraps(original_average_phases)
    def average_phases(phases: Any) -> np.ndarray:
        phases = band._materialize_numeric_input(phases)
        if _contains_complex_like(phases):
            raise ValueError(
                "phases must contain real-valued phase values, not complex values."
            )
        return original_average_phases(phases)

    band.validate_time_axis = validate_time_axis
    band.validate_sampling_rate = validate_sampling_rate
    band.validate_signal_values = validate_signal_values
    band.validate_band_hz = validate_band_hz
    band.circular_mean_phase = circular_mean_phase
    band.average_phases = average_phases
    setattr(band, _PATCH_MARKER, True)


__all__ = ["install"]
