"""Runtime patch for strict FieldTrip ``sampleinfo`` validation."""

from __future__ import annotations

from typing import Any

import numpy as np

_SAMPLEINFO_ERROR = "sampleinfo must contain finite integer sample bounds."
_PATCH_MARKER = "_neureptrace_fieldtrip_sampleinfo_validation_patched"


def _contains_boolean(value: np.ndarray) -> bool:
    """Return whether an array contains Python or NumPy boolean values."""

    if np.issubdtype(value.dtype, np.bool_):
        return True
    if value.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in value.ravel(order="C"))
    return False


def _as_integer_sample_bounds(array: np.ndarray) -> np.ndarray:
    """Validate and return integer FieldTrip sample-bound pairs."""

    if _contains_boolean(array):
        raise ValueError(_SAMPLEINFO_ERROR)

    if np.issubdtype(array.dtype, np.integer):
        return array.astype(int, copy=False)

    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_SAMPLEINFO_ERROR) from exc

    if not np.all(np.isfinite(numeric)):
        raise ValueError(_SAMPLEINFO_ERROR)

    rounded = np.round(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(_SAMPLEINFO_ERROR)
    return rounded.astype(int, copy=False)


def install() -> None:
    """Install strict FieldTrip ``sampleinfo`` integer validation."""

    import neureptrace.fieldtrip_mat as fieldtrip_mat

    if getattr(fieldtrip_mat._sampleinfo_array, _PATCH_MARKER, False):
        return

    def _sampleinfo_array(value: Any | None, *, n_trials: int) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(fieldtrip_mat._unwrap_scalar_object(value))
        if array.size == 0:
            return None
        if array.ndim == 1:
            if array.size != n_trials * 2:
                raise ValueError(f"sampleinfo must have shape {(n_trials, 2)}, got {array.shape}.")
            array = array.reshape(n_trials, 2)
        if array.ndim > 1 and array.shape[0] != n_trials and array.shape[-1] == n_trials:
            array = array.T
        if array.shape != (n_trials, 2):
            raise ValueError(f"sampleinfo must have shape {(n_trials, 2)}, got {array.shape}.")
        return _as_integer_sample_bounds(array)

    setattr(_sampleinfo_array, _PATCH_MARKER, True)
    fieldtrip_mat._sampleinfo_array = _sampleinfo_array


__all__ = ["install"]
