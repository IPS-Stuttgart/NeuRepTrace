"""Validate oscillatory feature channel indices before NumPy coercion."""

from __future__ import annotations

import functools
import importlib
from collections.abc import Iterable
from typing import Any

import numpy as np

_PATCH_ATTR = "_neureptrace_oscillatory_channel_indices_validated"
_INDEX_ERROR = "channel_indices must be a non-empty one-dimensional sequence of integers."


def _normalize_channel_indices(value: Any) -> np.ndarray:
    """Return platform-sized integer indices without lossy type coercion."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(_INDEX_ERROR)
    materialized = value if isinstance(value, np.ndarray) else tuple(value)
    try:
        array = np.asarray(materialized)
    except (TypeError, ValueError) as exc:
        raise ValueError(_INDEX_ERROR) from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(_INDEX_ERROR)
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(_INDEX_ERROR)

    if np.issubdtype(array.dtype, np.integer):
        indices = [int(item) for item in array.tolist()]
    elif array.dtype == object:
        indices = []
        for item in array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError(_INDEX_ERROR)
            indices.append(int(item))
    else:
        raise ValueError(_INDEX_ERROR)

    index_info = np.iinfo(np.intp)
    if any(item < index_info.min or item > index_info.max for item in indices):
        raise ValueError("channel_indices contain a value outside the platform index range.")
    return np.asarray(indices, dtype=np.intp)


def _validated_channel_indices(original):
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        channel_indices = kwargs.get("channel_indices")
        if channel_indices is not None:
            kwargs = dict(kwargs)
            kwargs["channel_indices"] = _normalize_channel_indices(channel_indices)
        return original(*args, **kwargs)

    setattr(wrapped, _PATCH_ATTR, True)
    return wrapped


def install() -> None:
    """Install channel-index validation on both oscillatory feature APIs."""

    oscillatory = importlib.import_module("neureptrace.features.oscillatory")
    for name in ("compute_band_trial_features", "compute_band_features"):
        current = getattr(oscillatory, name)
        if not getattr(current, _PATCH_ATTR, False):
            setattr(oscillatory, name, _validated_channel_indices(current))


__all__ = ["install"]
