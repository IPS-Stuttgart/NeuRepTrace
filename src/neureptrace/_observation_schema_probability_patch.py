"""Runtime probability-domain guards for probability observations.

Strict schema checks live directly in ``neureptrace.observation_schema``.  This
install hook remains so older package initialization paths can keep calling it,
and it also guards ``ProbabilityObservationTable.from_decoded_fold`` so
workflows fail before emitting invalid probability-observation rows.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from neureptrace.observation_schema import DEFAULT_PROBABILITY_TOLERANCE

_PATCH_MARKER = "_neureptrace_observation_probability_patch_installed"
_DECODED_FOLD_PATCH_MARKER = "_neureptrace_from_decoded_fold_probability_patch_installed"


def _contains_boolean_values(values: Any) -> bool:
    """Return whether an array-like value contains Python or NumPy booleans."""

    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in array.ravel())
    return False


def _validate_decoded_fold_probabilities(probabilities: Any) -> None:
    """Reject invalid probability matrices before canonical rows are emitted."""

    if _contains_boolean_values(probabilities):
        raise ValueError("from_decoded_fold probabilities must be numeric, not boolean.")
    try:
        values = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("from_decoded_fold probabilities must be numeric.") from exc
    if values.ndim != 2:
        return
    if not np.isfinite(values).all():
        raise ValueError("from_decoded_fold probabilities must be finite.")
    if np.any(values < 0.0):
        raise ValueError("from_decoded_fold probabilities must be non-negative.")
    if np.any(values > 1.0):
        raise ValueError("from_decoded_fold probabilities must not exceed 1.0.")
    row_sums = values.sum(axis=1)
    bad_rows = np.flatnonzero(np.abs(row_sums - 1.0) > DEFAULT_PROBABILITY_TOLERANCE)
    if bad_rows.size:
        examples = [float(row_sums[index]) for index in bad_rows[:5]]
        raise ValueError(
            "from_decoded_fold probability rows must sum to 1.0 within tolerance "
            f"{DEFAULT_PROBABILITY_TOLERANCE:g}; example row sums: {examples}"
        )


def _validate_decoded_fold_integer_values(values: Any, *, name: str) -> None:
    """Reject boolean label/index vectors before NumPy coerces them to 0/1."""

    if _contains_boolean_values(values):
        raise ValueError(f"from_decoded_fold {name} must be integer-valued, not boolean.")


def _install_decoded_fold_probability_guard() -> None:
    """Install a classmethod wrapper for decoded-fold observation construction."""

    from neureptrace.observations import ProbabilityObservationTable

    if getattr(ProbabilityObservationTable, _DECODED_FOLD_PATCH_MARKER, False):
        return

    original_from_decoded_fold = ProbabilityObservationTable.from_decoded_fold.__func__

    @classmethod
    @wraps(original_from_decoded_fold)
    def from_decoded_fold(cls, *args: Any, **kwargs: Any):
        if "probabilities" in kwargs:
            _validate_decoded_fold_probabilities(kwargs["probabilities"])
        for name in ("test_labels", "predictions", "test_indices"):
            if name in kwargs:
                _validate_decoded_fold_integer_values(kwargs[name], name=name)
        return original_from_decoded_fold(cls, *args, **kwargs)

    ProbabilityObservationTable.from_decoded_fold = from_decoded_fold
    setattr(ProbabilityObservationTable, _DECODED_FOLD_PATCH_MARKER, True)


def install() -> None:
    """Mark the legacy schema patch and install decoded-fold probability guards."""

    from neureptrace import observation_schema

    setattr(observation_schema, _PATCH_MARKER, True)
    _install_decoded_fold_probability_guard()
