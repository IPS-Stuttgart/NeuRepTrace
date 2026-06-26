"""Reject boolean values for source-alignment time controls."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_alignment_times_validation_patch_installed"


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _boolean_time_error() -> ValueError:
    return ValueError("alignment_times must contain finite numeric time centers, not booleans.")


def _install_source_selection_temperature_patch() -> None:
    source_selection_temperature_patch = importlib.import_module("neureptrace._source_selection_temperature_patch")
    source_selection_temperature_patch.install()


def install() -> None:
    """Patch source-alignment time parsing to reject boolean scalars."""

    _install_source_selection_temperature_patch()

    source_alignment = importlib.import_module("neureptrace.decoding.source_alignment")
    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    original_parse_alignment_times = source_alignment.parse_alignment_times

    @wraps(original_parse_alignment_times)
    def parse_alignment_times(times: Sequence[float] | str | None) -> tuple[tuple[float, ...], bool]:
        if _is_bool_scalar(times):
            raise _boolean_time_error()
        if times is not None and not isinstance(times, (str, bytes)):
            try:
                values = tuple(times)
            except TypeError:
                return original_parse_alignment_times(times)
            if any(_is_bool_scalar(value) for value in values):
                raise _boolean_time_error()
            return original_parse_alignment_times(values)
        return original_parse_alignment_times(times)

    source_alignment.parse_alignment_times = parse_alignment_times
    setattr(source_alignment, _PATCH_MARKER, True)


__all__ = ["install"]
