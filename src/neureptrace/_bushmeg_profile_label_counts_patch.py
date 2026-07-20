"""Handle composite labels and class-count metadata in BUSH-MEG summaries."""

from __future__ import annotations

import importlib
from collections import Counter
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_bushmeg_profile_label_counts_patch_installed"


def _normalise_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalise_value(value.item())
        return tuple(_normalise_value(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_normalise_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalise_value(item) for item in value)
    return value


def _label_values(labels: Any) -> list[Any]:
    array = np.asarray(labels, dtype=object)
    if array.ndim == 0:
        return [_normalise_value(array.item())]
    if array.ndim == 1:
        return [_normalise_value(value) for value in array.tolist()]
    if array.ndim == 2 and array.shape[1] == 1:
        return [_normalise_value(value) for value in array[:, 0].tolist()]
    return [_normalise_value(tuple(row)) for row in array.reshape(array.shape[0], -1).tolist()]


def _count_key(value: Any) -> str:
    value = _normalise_value(value)
    return repr(value) if isinstance(value, tuple) else str(value)


def _class_count_dict(labels: Any) -> dict[str, int]:
    counts = Counter(_count_key(value) for value in _label_values(labels))
    return {key: int(counts[key]) for key in sorted(counts)}


def install() -> None:
    importlib.import_module("neureptrace._bushmeg_diagnostics_class_count_patch").install()

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return
    all_protocols._class_count_dict = _class_count_dict
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
