"""Runtime hardening patch for probability-observation validation.

NeuRepTrace's observation-schema validator is used as a guardrail before
probability tables feed temporal models and detection workflows. This patch
keeps the public validator API stable while rejecting impossible probability
entries that can otherwise pass through as row-sum warnings.
It can be folded directly into ``neureptrace.observation_schema`` later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_PATCH_MARKER = "_neureptrace_observation_probability_patch_installed"


def _finite_mask(values: pd.Series) -> pd.Series:
    """Return an index-aligned mask for finite numeric entries."""

    return pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=values.index)


def install() -> None:
    """Install strict probability-domain checks for observation validation."""

    from neureptrace import observation_schema

    if getattr(observation_schema, _PATCH_MARKER, False):
        return

    original_validate_probabilities = observation_schema._validate_probabilities

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

    _validate_probabilities.__doc__ = original_validate_probabilities.__doc__
    observation_schema._validate_probabilities = _validate_probabilities
    setattr(observation_schema, _PATCH_MARKER, True)
