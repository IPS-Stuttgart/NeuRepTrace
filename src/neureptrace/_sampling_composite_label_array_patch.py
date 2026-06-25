"""Preserve NumPy composite label rows in class-limiting sampling."""

from __future__ import annotations

import numpy as np

_PATCH_MARKER = "_neureptrace_sampling_composite_label_array_patch_installed"


def _label_vector(labels) -> np.ndarray:
    """Return labels as a one-dimensional object vector without splitting rows."""

    if isinstance(labels, np.ndarray):
        array = labels.astype(object, copy=False)
        if array.ndim == 0:
            return array.reshape(1)
        if array.ndim == 1:
            return array
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            return rows[:, 0].astype(object, copy=False)
        vector = np.empty(rows.shape[0], dtype=object)
        vector[:] = [tuple(row.tolist()) for row in rows]
        return vector

    if isinstance(labels, (str, bytes)):
        return np.asarray([labels], dtype=object)

    try:
        items = list(labels)
    except TypeError:
        items = [labels]

    if any(isinstance(item, tuple) for item in items):
        vector = np.empty(len(items), dtype=object)
        vector[:] = items
        return vector
    return np.asarray(items, dtype=object).reshape(-1)


def install() -> None:
    """Install a sampler label-vector fix for multi-column composite arrays."""

    from neureptrace.decoding import sampling

    if getattr(sampling, _PATCH_MARKER, False):
        return
    sampling._label_vector = _label_vector
    setattr(sampling, _PATCH_MARKER, True)


__all__ = ["install"]
