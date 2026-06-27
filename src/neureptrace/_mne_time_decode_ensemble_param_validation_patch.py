"""Runtime patch for strict time-decode ensemble parameter validation."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_WEIGHTS_ERROR = "logistic_svm_ensemble weights must be finite non-negative values with positive sum."
_TEMPERATURE_ERROR = "logistic_svm_ensemble source temperatures must be finite positive values."
_PARAM_PATCH_MARKER = "_neureptrace_ensemble_param_validation_patched"
_ARRAY_METADATA_PATCH_MARKER = "_neureptrace_ensemble_array_metadata_patched"
_ARRAY_METADATA_SEQUENCE_KWARGS = ("source_time_selection_times", "alignment_times")


def _contains_boolean(values: Sequence[Any]) -> bool:
    return any(isinstance(value, (bool, np.bool_)) for value in values)


def _coerce_numpy_metadata_sequence(value: Any) -> Any:
    if not isinstance(value, np.ndarray):
        return value
    if value.ndim == 0:
        return (value.item(),)
    return tuple(value.reshape(-1).tolist())


def _install_mne_time_decode_float_sequence_validation_patch() -> None:
    patch = importlib.import_module("neureptrace._mne_time_decode_float_sequence_validation_patch")
    patch.install()


def install() -> None:
    """Reject invalid ensemble options and normalize NumPy metadata kwargs."""

    _install_mne_time_decode_float_sequence_validation_patch()

    from neureptrace import mne_time_decode_ensemble as ensemble

    if not getattr(ensemble._parse_weights, _PARAM_PATCH_MARKER, False):
        original_parse_weights = ensemble._parse_weights

        @wraps(original_parse_weights)
        def _parse_weights(weights: Sequence[float] | None, n_sources: int) -> tuple[float, ...]:
            if weights is not None and _contains_boolean(weights):
                raise ValueError(_WEIGHTS_ERROR)
            return original_parse_weights(weights, n_sources)

        setattr(_parse_weights, _PARAM_PATCH_MARKER, True)
        ensemble._parse_weights = _parse_weights

    if not getattr(ensemble._parse_source_temperatures, _PARAM_PATCH_MARKER, False):
        original_parse_source_temperatures = ensemble._parse_source_temperatures

        @wraps(original_parse_source_temperatures)
        def _parse_source_temperatures(temperatures: Sequence[float] | None, n_sources: int) -> tuple[float, ...]:
            if temperatures is not None and _contains_boolean(temperatures):
                raise ValueError(_TEMPERATURE_ERROR)
            return original_parse_source_temperatures(temperatures, n_sources)

        setattr(_parse_source_temperatures, _PARAM_PATCH_MARKER, True)
        ensemble._parse_source_temperatures = _parse_source_temperatures

    if not getattr(ensemble.run_time_resolved_decode, _ARRAY_METADATA_PATCH_MARKER, False):
        original_run_time_resolved_decode = ensemble.run_time_resolved_decode

        @wraps(original_run_time_resolved_decode)
        def run_time_resolved_decode(*args: Any, **kwargs: Any):
            for key in _ARRAY_METADATA_SEQUENCE_KWARGS:
                if key in kwargs:
                    kwargs[key] = _coerce_numpy_metadata_sequence(kwargs[key])
            return original_run_time_resolved_decode(*args, **kwargs)

        setattr(run_time_resolved_decode, _ARRAY_METADATA_PATCH_MARKER, True)
        ensemble.run_time_resolved_decode = run_time_resolved_decode


__all__ = ["install"]
