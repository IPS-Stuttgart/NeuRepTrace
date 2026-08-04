"""Reject mapping containers in ordered numeric signal-processing inputs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable

_PATCH_MARKER = "_neureptrace_signal_mapping_input_patch_installed"
_ERROR = "Signal-processing numeric inputs must be ordered values, not mappings."


def _reject_mapping(value: Any) -> None:
    if isinstance(value, Mapping):
        raise ValueError(_ERROR)


def _guard_first_argument(original: Callable[..., Any]) -> Callable[..., Any]:
    """Reject a mapping before an API can iterate over its keys."""

    @wraps(original)
    def guarded(value: Any, *args: Any, **kwargs: Any) -> Any:
        _reject_mapping(value)
        return original(value, *args, **kwargs)

    setattr(guarded, _PATCH_MARKER, True)
    return guarded


def install() -> None:
    """Prevent dictionaries from being interpreted as sequences of numeric keys."""

    band = importlib.import_module("neureptrace.signal.band")

    original_materialize = band._materialize_numeric_input
    if not getattr(original_materialize, _PATCH_MARKER, False):

        @wraps(original_materialize)
        def _materialize_numeric_input(value: Any) -> object:
            _reject_mapping(value)
            return original_materialize(value)

        setattr(_materialize_numeric_input, _PATCH_MARKER, True)
        band._materialize_numeric_input = _materialize_numeric_input

    # These helpers consume their first iterable before calling the shared
    # materializer, so guard their public boundary explicitly as well.
    for function_name in ("validate_band_hz", "average_phases"):
        original = getattr(band, function_name)
        if not getattr(original, _PATCH_MARKER, False):
            setattr(band, function_name, _guard_first_argument(original))


__all__ = ["install"]
