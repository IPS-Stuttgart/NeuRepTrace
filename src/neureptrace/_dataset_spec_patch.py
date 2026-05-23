"""Runtime robustness patch for dataset-spec MATLAB time axes.

FieldTrip exports can store a time axis as a MATLAB row or column vector.  The
base dataset-spec loader already handled row vectors, but column vectors could
be reduced to their first sample by the generic 2-D fallback.  This patch keeps
that public loader behavior stable while preserving vector-shaped time axes.
It can be folded directly into ``neureptrace.dataset_spec`` later.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_dataset_spec_patch_installed"


def install() -> None:
    """Install robust MATLAB time-axis handling for dataset specs."""

    from neureptrace import dataset_spec

    if getattr(dataset_spec, _PATCH_MARKER, False):
        return

    sequence_from_mat_value = dataset_spec._sequence_from_mat_value
    unwrap_mat_object = dataset_spec._unwrap_mat_object
    original_first_time_axis = dataset_spec._first_time_axis

    def first_time_axis(times: Any) -> np.ndarray:
        items = sequence_from_mat_value(times)
        time_axis = np.asarray(unwrap_mat_object(items[0]), dtype=float)
        if time_axis.ndim == 0:
            time_axis = time_axis.reshape(1)
        elif time_axis.ndim == 1 or 1 in time_axis.shape:
            time_axis = time_axis.reshape(-1)
        else:
            time_axis = time_axis.reshape(time_axis.shape[0], -1)[0]
        if time_axis.size == 0:
            raise ValueError("MATLAB time axis is empty.")
        return time_axis

    first_time_axis.__doc__ = original_first_time_axis.__doc__
    dataset_spec._first_time_axis = first_time_axis
    setattr(dataset_spec, _PATCH_MARKER, True)
