"""Runtime patch for singleton condition alignment in exact observation aggregation."""

from __future__ import annotations

import importlib

import pandas as pd

_PATCH_ATTR = "_neureptrace_results_observation_condition_singletons"


def _explicit_unique_values(values: pd.Series) -> list[object]:
    non_missing = values.dropna()
    if non_missing.empty:
        return []
    non_blank = non_missing.loc[non_missing.astype(str).str.strip().ne("")]
    return list(pd.unique(non_blank))


def _has_explicit_values(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and bool(_explicit_unique_values(frame[column]))


def _prepare_observations_for_subject_time(
    results_frame: pd.DataFrame,
    observations: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Fill observation condition columns from unambiguous result singletons."""

    results = importlib.import_module("neureptrace.results")
    prepared = results._normalize_emission_mode(observations).copy()
    for column in group_columns:
        if _has_explicit_values(observations, column):
            continue

        values = _explicit_unique_values(results_frame[column])
        if len(values) == 1:
            prepared[column] = values[0]
            continue

        raise ValueError(
            "Probability observations are missing condition column "
            f"'{column}' while result metrics contain multiple or no explicit {column} values."
        )
    return prepared


def install() -> None:
    """Install singleton-aware condition alignment for exact observation metrics."""

    results = importlib.import_module("neureptrace.results")
    original = results._prepare_observations_for_subject_time
    if getattr(original, _PATCH_ATTR, False):
        return

    setattr(_prepare_observations_for_subject_time, _PATCH_ATTR, True)
    _prepare_observations_for_subject_time.__wrapped__ = original  # type: ignore[attr-defined]
    results._prepare_observations_for_subject_time = _prepare_observations_for_subject_time


__all__ = ["install"]
