"""Reject mapping containers in ordered numeric signal-processing inputs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_signal_mapping_input_patch_installed"


def install() -> None:
    """Prevent dictionaries from being interpreted as sequences of numeric keys."""

    band = importlib.import_module("neureptrace.signal.band")
    original_materialize = band._materialize_numeric_input
    if getattr(original_materialize, _PATCH_MARKER, False):
        return

    @wraps(original_materialize)
    def _materialize_numeric_input(value: Any) -> object:
        if isinstance(value, Mapping):
            raise ValueError(
                "Signal-processing numeric inputs must be ordered values, not mappings."
            )
        return original_materialize(value)

    setattr(_materialize_numeric_input, _PATCH_MARKER, True)
    band._materialize_numeric_input = _materialize_numeric_input


__all__ = ["install"]
