"""Runtime guard for single oscillatory feature window specs."""

from __future__ import annotations

import functools
import importlib
from collections.abc import Mapping
from typing import Any

_PATCH_ATTR = "_neureptrace_oscillatory_single_window"


def _is_single_window_sequence(value: Any, window_cls: type) -> bool:
    """Return True when ``value`` is one window spec, not a list of specs."""

    if isinstance(value, (window_cls, Mapping, str, bytes)):
        return False
    try:
        values = tuple(value)
    except TypeError:
        return False

    if len(values) == 2:
        return not any(_is_window_collection_item(item, window_cls) for item in values)
    if len(values) == 3 and isinstance(values[0], (str, bytes)):
        return not any(_is_window_collection_item(item, window_cls) for item in values[1:])
    return False


def _is_window_collection_item(value: Any, window_cls: type) -> bool:
    return isinstance(value, (window_cls, Mapping)) or _is_single_window_sequence(value, window_cls)


def install() -> None:
    oscillatory = importlib.import_module("neureptrace.features.oscillatory")
    original = getattr(oscillatory, "_normalize_windows")
    if getattr(original, _PATCH_ATTR, False):
        return

    window_cls = oscillatory.BandFeatureWindow

    @functools.wraps(original)
    def _normalize_windows(windows: Any):
        if _is_single_window_sequence(windows, window_cls):
            return (oscillatory._normalize_window(windows),)
        return original(windows)

    setattr(_normalize_windows, _PATCH_ATTR, True)
    oscillatory._normalize_windows = _normalize_windows


__all__ = ["install"]
