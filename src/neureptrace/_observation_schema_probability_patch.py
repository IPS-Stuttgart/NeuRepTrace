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
import pandas as pd

from neureptrace.observation_schema import DEFAULT_PROBABILITY_TOLERANCE

_PATCH_MARKER = "_neureptrace_observation_probability_patch_installed"
_PROBABILITY_VALIDATION_PATCH_MARKER = "_neureptrace_observation_partial_probability_sum_patch_installed"
_DECODED_FOLD_PATCH_MARKER = "_neureptrace_from_decoded_fold_probability_patch_installed"


def _contains_boolean_values(values: Any) -> bool:
    """Return whether an array-like value contains Python or NumPy booleans.

    Object arrays can contain zero-dimensional NumPy arrays or nested arrays as
    cells.  NumPy can coerce those boolean cells to 0/1 during a later
    ``astype(float)``/``astype(int)`` conversion, so inspect them recursively
    instead of only checking the top-level object-array elements.
    """

    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return True
        if values.dtype == object:
            return any(_contains_boolean_values(value) for value in values.ravel(order="C"))
        return False
    if isinstance(values, (str, bytes)):
        return False
    if isinstance(values, (list, tuple)):
        return any(_contains_boolean_values(value) for value in values)
    if hasattr(values, "__array__"):
        try:
            return _contains_boolean_values(np.asarray(values))
        except (TypeError, ValueError):
            return False
    return False


def _validate_observation_probabilities(
    probabilities: pd.DataFrame,
    issues: list[Any],
    *,
    tolerance: float,
    require_normalized: bool,
) -> None:
    """Validate probability-domain values while counting partial NaNs as zero mass.

    ``observation_schema._probability_frame`` allows empty probability cells so
    sparse exports can be inspected.  The row-sum check still needs to include
    those rows, treating missing cells as zero probability.  The pre-patch
    implementation used ``np.isfinite(...).all(axis=1)`` before summing, which
    also filtered out rows containing NaN and let partial probability rows bypass
    normalization diagnostics entirely.
    """

    if probabilities.empty:
        return

    from neureptrace import observation_schema

    for column in probabilities.columns:
        values = probabilities[column]
        finite = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=values.index)
        present_and_finite = values.notna() & finite

        non_finite_mask = values.notna() & ~finite
        for row_index, value in probabilities.loc[non_finite_mask, column].head(20).items():
            observation_schema._issue(
                issues,
                "error",
                "non_finite_probability",
                f"Probability column '{column}' must contain finite values when present.",
                column=column,
                row=int(row_index),
                value=float(value),
            )

        negative_mask = present_and_finite & (values < 0.0)
        for row_index, value in probabilities.loc[negative_mask, column].head(20).items():
            observation_schema._issue(
                issues,
                "error",
                "negative_probability",
                f"Probability column '{column}' contains a negative value.",
                column=column,
                row=int(row_index),
                value=float(value),
            )

        above_one_mask = present_and_finite & (values > 1.0)
        for row_index, value in probabilities.loc[above_one_mask, column].head(20).items():
            observation_schema._issue(
                issues,
                "error",
                "probability_above_one",
                f"Probability column '{column}' contains a value above 1.0.",
                column=column,
                row=int(row_index),
                value=float(value),
            )

    all_missing = probabilities.isna().all(axis=1)
    for row_index in probabilities.loc[all_missing].head(20).index:
        observation_schema._issue(
            issues,
            "error",
            "missing_probability_row",
            "Each observation row must contain at least one probability value.",
            row=int(row_index),
        )

    valid = probabilities.dropna(how="all")
    if valid.empty:
        return

    valid_values = valid.to_numpy(dtype=float)
    present_values = valid.notna().to_numpy(dtype=bool)
    non_finite_present = present_values & ~np.isfinite(valid_values)
    finite_rows = pd.Series(~non_finite_present.any(axis=1), index=valid.index)
    valid = valid.loc[finite_rows]
    if valid.empty:
        return

    row_sums = valid.fillna(0.0).sum(axis=1)
    deviations = (row_sums - 1.0).abs()
    bad_sums = deviations > tolerance
    severity = "error" if require_normalized else "warning"
    code = "probability_sum_error" if require_normalized else "probability_sum_warning"
    for row_index, _deviation in deviations.loc[bad_sums].head(20).items():
        observation_schema._issue(
            issues,
            severity,
            code,
            f"Probability row sums should be 1.0 within tolerance {tolerance:g}.",
            row=int(row_index),
            value=float(row_sums.loc[row_index]),
        )
    if int(bad_sums.sum()) > 20:
        observation_schema._issue(
            issues,
            severity,
            f"{code}_truncated",
            f"{int(bad_sums.sum())} probability rows have sums outside tolerance {tolerance:g}; first 20 are listed.",
        )


def _install_probability_validation_guard() -> None:
    """Install the partial-missing probability row-sum validator."""

    from neureptrace import observation_schema

    if getattr(observation_schema, _PROBABILITY_VALIDATION_PATCH_MARKER, False):
        return
    observation_schema._validate_probabilities = _validate_observation_probabilities
    setattr(observation_schema, _PROBABILITY_VALIDATION_PATCH_MARKER, True)


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
    """Mark the legacy schema patch and install probability-domain guards."""

    from neureptrace import observation_schema

    setattr(observation_schema, _PATCH_MARKER, True)
    _install_probability_validation_guard()
    _install_decoded_fold_probability_guard()
