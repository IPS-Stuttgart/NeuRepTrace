"""Runtime guard for oscillatory feature labels with composite values."""

from __future__ import annotations

import functools
import importlib
from typing import Any

import numpy as np

_PATCH_ATTR = "_neureptrace_oscillatory_composite_labels"


def _label_cell_value(value: Any) -> Any:
    """Return a stable scalar-or-tuple label value for one trial."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _label_cell_value(value.item())
        flattened = [_label_cell_value(item) for item in value.ravel().tolist()]
        return flattened[0] if len(flattened) == 1 else tuple(flattened)
    if isinstance(value, tuple):
        return tuple(_label_cell_value(item) for item in value)
    if isinstance(value, list):
        flattened = [_label_cell_value(item) for item in value]
        return flattened[0] if len(flattened) == 1 else tuple(flattened)
    return value


def _normalize_trial_labels(labels: Any, n_trials: int) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=object)
    if labels_array.ndim == 0 or labels_array.shape[0] != int(n_trials):
        raise ValueError("labels must contain one value per trial.")

    normalized = np.empty(int(n_trials), dtype=object)
    for trial_idx in range(int(n_trials)):
        normalized[trial_idx] = _label_cell_value(labels_array[trial_idx])
    return normalized


def install() -> None:
    oscillatory = importlib.import_module("neureptrace.features.oscillatory")
    original = getattr(oscillatory, "compute_band_features")
    if getattr(original, _PATCH_ATTR, False):
        return

    @functools.wraps(original)
    def compute_band_features(data: Any, time_vector: Any, *, labels: Any = None, **kwargs: Any):
        if labels is not None:
            data_array = np.asarray(data, dtype=float)
            if data_array.ndim == 3:
                trial_axis = oscillatory._normalize_axis(kwargs.get("trial_axis", 0), data_array.ndim)
                labels = _normalize_trial_labels(labels, int(data_array.shape[trial_axis]))
        return original(data, time_vector, labels=labels, **kwargs)

    setattr(compute_band_features, _PATCH_ATTR, True)
    oscillatory.compute_band_features = compute_band_features


__all__ = ["install"]
