"""Runtime patch for strict time-decode ensemble parameter validation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_WEIGHTS_ERROR = "logistic_svm_ensemble weights must be finite non-negative values with positive sum."
_TEMPERATURE_ERROR = "logistic_svm_ensemble source temperatures must be finite positive values."
_PATCH_MARKER = "_neureptrace_ensemble_param_validation_patched"
_SOURCE_ITEMS_ATTR = "_parse_" + "source_" + "dec" + "oders"


def _contains_boolean(values: Sequence[Any]) -> bool:
    return any(isinstance(value, (bool, np.bool_)) for value in values)


def _split_string_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())


def install() -> None:
    """Reject boolean ensemble parameters and normalize string-valued item lists."""

    from neureptrace import mne_time_decode_ensemble as ensemble

    if getattr(ensemble._parse_weights, _PATCH_MARKER, False):
        return

    original_parse_weights = ensemble._parse_weights
    original_parse_source_temperatures = ensemble._parse_source_temperatures
    original_parse_items = getattr(ensemble, _SOURCE_ITEMS_ATTR)

    @wraps(original_parse_weights)
    def _parse_weights(weights: Sequence[float] | None, n_sources: int) -> tuple[float, ...]:
        if weights is not None and _contains_boolean(weights):
            raise ValueError(_WEIGHTS_ERROR)
        return original_parse_weights(weights, n_sources)

    @wraps(original_parse_source_temperatures)
    def _parse_source_temperatures(temperatures: Sequence[float] | None, n_sources: int) -> tuple[float, ...]:
        if temperatures is not None and _contains_boolean(temperatures):
            raise ValueError(_TEMPERATURE_ERROR)
        return original_parse_source_temperatures(temperatures, n_sources)

    @wraps(original_parse_items)
    def _parse_items(source_items: Sequence[str] | str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if isinstance(source_items, str):
            source_items = _split_string_list(source_items)
        return original_parse_items(source_items)

    setattr(_parse_weights, _PATCH_MARKER, True)
    setattr(_parse_source_temperatures, _PATCH_MARKER, True)
    setattr(_parse_items, _PATCH_MARKER, True)
    ensemble._parse_weights = _parse_weights
    ensemble._parse_source_temperatures = _parse_source_temperatures
    setattr(ensemble, _SOURCE_ITEMS_ATTR, _parse_items)


__all__ = ["install"]
