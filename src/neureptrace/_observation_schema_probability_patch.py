"""Runtime hardening patch for probability-observation validation.

NeuRepTrace's observation-schema validator is used as a guardrail before
probability tables feed temporal models and detection workflows. This patch
keeps the public validator API stable while rejecting impossible probability
entries that can otherwise pass through as row-sum warnings. It also rejects
non-finite tolerances because they make row-sum and consistency checks
ill-defined.
It can be folded directly into ``neureptrace.observation_schema`` later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_PATCH_MARKER = "_neureptrace_observation_probability_patch_installed"
_TOLERANCE_ERROR = "probability_tolerance must be finite and non-negative."


def _finite_mask(values: pd.Series) -> pd.Series:
    """Return an index-aligned mask for finite numeric entries."""

    return pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=values.index)


def _validate_probability_tolerance(probability_tolerance: float) -> None:
    """Reject tolerances that would disable numeric validation checks."""

    try:
        tolerance = float(probability_tolerance)
    except (TypeError, ValueError):
        raise ValueError(_TOLERANCE_ERROR) from None
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError(_TOLERANCE_ERROR)


def install() -> None:
    """Install strict probability-domain checks for observation validation."""

    from neureptrace import observation_schema

    if getattr(observation_schema, _PATCH_MARKER, False):
        return

    original_validate_probability_observations = observation_schema.validate_probability_observations
    original_validate_probabilities = observation_schema._validate_probabilities

    def validate_probability_observations(
        frame: pd.DataFrame,
        *,
        profile: observation_schema.ObservationProfile = "generic",
        probability_tolerance: float = observation_schema.DEFAULT_PROBABILITY_TOLERANCE,
        require_normalized: bool = False,
        group_columns: list[str] | tuple[str, ...] | None = None,
        stream_columns: list[str] | tuple[str, ...] | None = None,
    ) -> observation_schema.ObservationValidationReport:
        _validate_probability_tolerance(probability_tolerance)
        return original_validate_probability_observations(
            frame,
            profile=profile,
            probability_tolerance=probability_tolerance,
            require_normalized=require_normalized,
            group_columns=group_columns,
            stream_columns=stream_columns,
        )

    def _validate_probabilities(
        probabilities: pd.DataFrame,
        issues: list[observation_schema.ObservationValidationIssue],
        *,
        tolerance: float,
        require_normalized: bool,
    ) -> None:
        original_validate_probabilities(
            probabilities,
            issues,
            tolerance=tolerance,
            require_normalized=require_normalized,
        )
        if probabilities.empty:
            return

        for column in probabilities.columns:
            values = probabilities[column]
            present = values.notna()
            finite = _finite_mask(values)

            non_finite_mask = present & ~finite
            for row_index, value in values.loc[non_finite_mask].head(20).items():
                observation_schema._issue(
                    issues,
                    "error",
                    "non_finite_probability",
                    f"Probability column '{column}' must contain finite values.",
                    column=column,
                    row=int(row_index),
                    value=float(value),
                )
            if int(non_finite_mask.sum()) > 20:
                observation_schema._issue(
                    issues,
                    "error",
                    "non_finite_probability_truncated",
                    f"Probability column '{column}' contains {int(non_finite_mask.sum())} non-finite values; first 20 are listed.",
                    column=column,
                )

            above_one_mask = present & finite & (values > 1.0)
            for row_index, value in values.loc[above_one_mask].head(20).items():
                observation_schema._issue(
                    issues,
                    "error",
                    "probability_above_one",
                    f"Probability column '{column}' contains a value above 1.0.",
                    column=column,
                    row=int(row_index),
                    value=float(value),
                )
            if int(above_one_mask.sum()) > 20:
                observation_schema._issue(
                    issues,
                    "error",
                    "probability_above_one_truncated",
                    f"Probability column '{column}' contains {int(above_one_mask.sum())} values above 1.0; first 20 are listed.",
                    column=column,
                )

    validate_probability_observations.__doc__ = original_validate_probability_observations.__doc__
    _validate_probabilities.__doc__ = original_validate_probabilities.__doc__
    observation_schema.validate_probability_observations = validate_probability_observations
    observation_schema._validate_probabilities = _validate_probabilities
    setattr(observation_schema, _PATCH_MARKER, True)
