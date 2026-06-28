"""Runtime patch for singleton condition alignment in exact observation aggregation."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

_PREPARE_PATCH_ATTR = "_neureptrace_results_observation_condition_singletons"
_READ_PATCH_ATTR = "_neureptrace_results_observation_condition_attrs"
_EXPLICIT_COLUMNS_ATTR = "_neureptrace_explicit_observation_columns"


def _explicit_unique_values(values: pd.Series) -> list[object]:
    non_missing = values.dropna()
    if non_missing.empty:
        return []
    non_blank = non_missing.loc[non_missing.astype(str).str.strip().ne("")]
    return list(pd.unique(non_blank))


def _has_explicit_values(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and bool(_explicit_unique_values(frame[column]))


def _explicit_observation_columns(observations: pd.DataFrame) -> set[str]:
    columns = observations.attrs.get(_EXPLICIT_COLUMNS_ATTR)
    if columns is None:
        return set(observations.columns)
    return {str(column) for column in columns}


def _header_columns(csv_paths: list[Path]) -> set[str]:
    columns: set[str] = set()
    for csv_path in csv_paths:
        columns.update(str(column) for column in pd.read_csv(csv_path, nrows=0).columns)
    return columns


def _prepare_observations_for_subject_time(
    results_frame: pd.DataFrame,
    observations: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Fill observation condition columns from unambiguous result singletons."""

    results = importlib.import_module("neureptrace.results")
    prepared = results._normalize_emission_mode(observations).copy()
    explicit_columns = _explicit_observation_columns(observations)
    for column in group_columns:
        if column in explicit_columns and _has_explicit_values(observations, column):
            continue

        values = _explicit_unique_values(results_frame[column])
        if len(values) == 1:
            prepared[column] = values[0]
            continue
        singleton_values = results_frame[column].dropna().astype(str).unique()
        if len(singleton_values) == 1:
            prepared[column] = singleton_values[0]
            continue

        raise ValueError(
            "Probability observations are missing condition column "
            f"'{column}' while result metrics contain multiple or no explicit {column} values."
        )
    return prepared


def install() -> None:
    """Install singleton-aware condition alignment for exact observation metrics."""

    results = importlib.import_module("neureptrace.results")
    original_prepare = results._prepare_observations_for_subject_time
    if not getattr(original_prepare, _PREPARE_PATCH_ATTR, False):
        setattr(_prepare_observations_for_subject_time, _PREPARE_PATCH_ATTR, True)
        _prepare_observations_for_subject_time.__wrapped__ = original_prepare  # type: ignore[attr-defined]
        results._prepare_observations_for_subject_time = _prepare_observations_for_subject_time

    original_read = results.read_probability_observations
    if getattr(original_read, _READ_PATCH_ATTR, False):
        return

    def read_probability_observations_with_condition_attrs(
        csv_paths: list[Path],
        *,
        subject_column: str | None = None,
        fallback_subjects_by_file: Mapping[str, object] | None = None,
    ) -> pd.DataFrame:
        observations = original_read(
            csv_paths,
            subject_column=subject_column,
            fallback_subjects_by_file=fallback_subjects_by_file,
        )
        observations.attrs[_EXPLICIT_COLUMNS_ATTR] = _header_columns(csv_paths)
        return observations

    setattr(read_probability_observations_with_condition_attrs, _READ_PATCH_ATTR, True)
    read_probability_observations_with_condition_attrs.__wrapped__ = original_read  # type: ignore[attr-defined]
    results.read_probability_observations = read_probability_observations_with_condition_attrs


__all__ = ["install"]
