"""Runtime hardening patch for observation label validation.

Canonical probability-observation tables use integer label columns to connect
``true_label``/``predicted_label`` values with ``prob_class_<label>`` and
``class_<label>`` columns.  The base validator historically cast these labels
with ``int(...)`` inside consistency checks, which could silently truncate
fractional labels or crash on non-finite values.  This patch keeps the public
validator API stable while rejecting malformed label values before they reach
those casts.
It can be folded directly into ``neureptrace.observation_schema`` later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_PATCH_MARKER = "_neureptrace_observation_label_patch_installed"
_LABEL_COLUMNS = ("predicted_label", "true_label")


def _invalid_label_mask(values: pd.Series) -> pd.Series:
    """Return an index-aligned mask for invalid present label values."""

    numeric = pd.to_numeric(values, errors="coerce")
    raw = numeric.to_numpy(dtype=float)
    finite = np.isfinite(raw)
    valid = finite & (raw >= 0.0) & (raw == np.floor(raw))
    return numeric.notna() & ~pd.Series(valid, index=values.index)


def _sanitize_label_columns(frame: pd.DataFrame, issues: list, *, observation_schema) -> pd.DataFrame:
    """Report and mask malformed integer label columns before base checks run."""

    present_columns = [column for column in _LABEL_COLUMNS if column in frame.columns]
    if not present_columns:
        return frame

    sanitized = frame.copy()
    for column in present_columns:
        numeric = pd.to_numeric(sanitized[column], errors="coerce")
        invalid = _invalid_label_mask(sanitized[column])
        for row_index, value in numeric.loc[invalid].head(20).items():
            observation_schema._issue(
                issues,
                "error",
                f"invalid_{column}",
                f"Column '{column}' must contain finite, non-negative integer class labels when present.",
                column=column,
                row=int(row_index),
                value=float(value),
            )
        if int(invalid.sum()) > 20:
            observation_schema._issue(
                issues,
                "error",
                f"invalid_{column}_truncated",
                f"Column '{column}' contains {int(invalid.sum())} invalid label values; first 20 are listed.",
                column=column,
            )
        sanitized.loc[invalid, column] = np.nan
    return sanitized


def install() -> None:
    """Install strict integer-label checks for canonical observation validation."""

    from neureptrace import observation_schema

    if getattr(observation_schema, _PATCH_MARKER, False):
        return

    original_validate_probability_consistency = observation_schema._validate_probability_consistency

    def _validate_probability_consistency(
        frame: pd.DataFrame,
        probabilities: pd.DataFrame,
        prob_columns,
        issues: list[observation_schema.ObservationValidationIssue],
        *,
        tolerance: float,
    ) -> None:
        sanitized = _sanitize_label_columns(frame, issues, observation_schema=observation_schema)
        original_validate_probability_consistency(
            sanitized,
            probabilities,
            prob_columns,
            issues,
            tolerance=tolerance,
        )

    _validate_probability_consistency.__doc__ = original_validate_probability_consistency.__doc__
    observation_schema._validate_probability_consistency = _validate_probability_consistency
    setattr(observation_schema, _PATCH_MARKER, True)
