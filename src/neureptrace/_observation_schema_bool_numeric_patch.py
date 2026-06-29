"""Reject boolean values in probability-observation numeric fields.

Pandas treats booleans as numeric during ``to_numeric`` coercion. Without this
check, malformed observation tables can silently turn ``True``/``False`` into
``1``/``0`` for time, probability, confidence, and provenance columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_observation_schema_bool_numeric_patch_installed"


def _boolean_mask(values: pd.Series) -> pd.Series:
    return values.map(lambda value: isinstance(value, (bool, np.bool_))).fillna(False).astype(bool)


def _make_numeric_series_checked(observation_schema):
    def _numeric_series_checked(
        frame: pd.DataFrame,
        column: str,
        issues: list,
        *,
        allow_nan: bool = False,
    ) -> pd.Series:
        raw_values = frame[column]
        boolean_values = _boolean_mask(raw_values)
        for row_index, value in raw_values.loc[boolean_values].head(20).items():
            observation_schema._issue(
                issues,
                "error",
                "boolean_numeric_value",
                f"Column '{column}' contains a boolean value; use an explicit numeric value instead.",
                column=column,
                row=int(row_index),
                value=value,
            )
        if int(boolean_values.sum()) > 20:
            observation_schema._issue(
                issues,
                "error",
                "boolean_numeric_value_truncated",
                f"Column '{column}' contains {int(boolean_values.sum())} boolean values; first 20 are listed.",
                column=column,
            )

        numeric_source = raw_values.mask(boolean_values, np.nan)
        values = pd.to_numeric(numeric_source, errors="coerce")
        bad_mask = values.isna() if not allow_nan else values.isna() & raw_values.notna()
        bad_mask = bad_mask & ~boolean_values
        for row_index, value in raw_values.loc[bad_mask].head(20).items():
            observation_schema._issue(
                issues,
                "error",
                "non_numeric_value",
                f"Column '{column}' contains a non-numeric value.",
                column=column,
                row=int(row_index),
                value=value,
            )
        if int(bad_mask.sum()) > 20:
            observation_schema._issue(
                issues,
                "error",
                "non_numeric_value_truncated",
                f"Column '{column}' contains {int(bad_mask.sum())} non-numeric values; first 20 are listed.",
                column=column,
            )
        return values

    return _numeric_series_checked


def install() -> None:
    """Install observation-schema numeric boolean rejection."""

    from neureptrace import observation_schema

    if getattr(observation_schema, _PATCH_MARKER, False):
        return
    observation_schema._numeric_series = _make_numeric_series_checked(observation_schema)
    setattr(observation_schema, _PATCH_MARKER, True)
