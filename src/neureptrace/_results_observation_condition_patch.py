"""Runtime patches for result aggregation observation and scalar-option handling."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

_PREPARE_PATCH_ATTR = "_neureptrace_results_observation_condition_singletons"
_READ_PATCH_ATTR = "_neureptrace_results_observation_condition_attrs"
_EXPLICIT_COLUMNS_ATTR = "_neureptrace_explicit_observation_columns"
_POSITIVE_INTEGER_ARRAYLIKE_ATTR = "_neureptrace_results_rejects_arraylike_positive_integer_controls"
_FINITE_SCALAR_ARRAYLIKE_ATTR = "_neureptrace_results_rejects_arraylike_finite_numeric_scalar_controls"
_COMPLEX_METRIC_TABLE_ATTR = "_neureptrace_results_rejects_complex_metric_table_values"
_EXACT_OBSERVATION_LABEL_ATTR = "_neureptrace_results_exact_probability_observation_labels"
_SIGNED_PROBABILITY_LABEL_RE = re.compile(r"^[+-]?\d+$")
_MAX_EXACT_FLOAT_INTEGER = 2**53


def _is_arraylike_scalar_control(value: object) -> bool:
    return isinstance(value, (np.ndarray, pd.Series, pd.Index, pd.api.extensions.ExtensionArray))


def _is_boolean_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return isinstance(value.item(), (bool, np.bool_))
    return False


def _boolean_rows(values: pd.Series) -> list[object]:
    mask = values.map(_is_boolean_scalar).fillna(False).astype(bool)
    return mask[mask].index.tolist()[:5]


def _is_complex_scalar(value: object) -> bool:
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return isinstance(value.item(), (complex, np.complexfloating))
    return False


def _complex_rows(values: pd.Series) -> list[object]:
    mask = values.map(_is_complex_scalar).fillna(False).astype(bool)
    return mask[mask].index.tolist()[:5]


def _column_names(columns: Sequence[str] | str | None) -> tuple[str, ...]:
    if columns is None:
        return ()
    if isinstance(columns, str):
        return (columns,)
    return tuple(dict.fromkeys(columns))


def _exact_integer_label(value: object, *, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} values must be scalar integer labels.")
        value = value.item()

    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must not contain booleans.")

    if value is None:
        raise ValueError(f"{name} must be numeric and non-missing.")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be numeric and non-missing.")
        if not numeric.is_integer():
            raise ValueError("Probability-observation true_label values must be integer-valued.")
        if abs(numeric) > _MAX_EXACT_FLOAT_INTEGER:
            raise ValueError(
                "Probability-observation true_label values must be exact integer labels; "
                "values above 2**53 must be supplied as integers or decimal strings."
            )
        return int(numeric)

    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be numeric and non-missing.")
    try:
        numeric = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and non-missing.") from exc
    if not numeric.is_finite():
        raise ValueError(f"{name} must be numeric and non-missing.")
    integral = numeric.to_integral_value()
    if numeric != integral:
        raise ValueError("Probability-observation true_label values must be integer-valued.")
    return int(integral)


def _exact_integer_labels(values: pd.Series) -> np.ndarray:
    name = "Probability-observation column 'true_label'"
    labels = np.empty(len(values), dtype=object)
    for index, value in enumerate(values.tolist()):
        labels[index] = _exact_integer_label(value, name=name)
    return labels


def _probability_label_values(prob_columns: Sequence[str]) -> tuple[int, ...]:
    suffixes = tuple(column.removeprefix("prob_class_") for column in prob_columns)
    if not all(_SIGNED_PROBABILITY_LABEL_RE.fullmatch(suffix) for suffix in suffixes):
        return tuple(range(len(prob_columns)))

    labels = tuple(int(suffix) for suffix in suffixes)
    seen: set[int] = set()
    duplicates: list[int] = []
    for label in labels:
        if label in seen and label not in duplicates:
            duplicates.append(label)
        seen.add(label)
    if duplicates:
        raise ValueError(
            "prob_class_* columns must map to unique class labels; "
            f"duplicate label(s): {duplicates}."
        )
    return labels


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


def _probability_ece_by_group_exact(
    observations: pd.DataFrame,
    group_columns: list[str],
    *,
    n_bins: int,
) -> pd.DataFrame:
    """Compute grouped ECE without routing exact integer labels through float64."""

    results = importlib.import_module("neureptrace.results")
    prob_columns = list(results.probability_columns(observations))
    missing = [column for column in (*group_columns, "true_label") if column not in observations.columns]
    if not prob_columns:
        missing.append("prob_class_*")
    if missing:
        raise ValueError(f"Probability observations are missing required columns: {missing}")

    for column in ("time", "true_label", *prob_columns):
        if column not in observations.columns:
            continue
        rows = _boolean_rows(observations[column])
        if rows:
            raise ValueError(f"Probability-observation column '{column}' must not contain booleans; bad row(s): {rows}.")

    working = observations.copy()
    working["time"] = pd.to_numeric(working["time"], errors="coerce")
    if working["time"].isna().any():
        raise ValueError("Probability-observation column 'time' must be numeric and non-missing.")

    labels = _exact_integer_labels(working["true_label"])
    working["true_label"] = pd.Series(labels.tolist(), index=working.index, dtype=object)

    for column in prob_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if working[prob_columns].isna().any().any():
        raise ValueError("Probability-observation columns must be numeric and non-missing.")

    probability_label_values = _probability_label_values(prob_columns)
    results._label_positions(labels, probability_label_values)
    probabilities = working[prob_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability-observation columns must be finite.")

    rows: list[dict[str, object]] = []
    for group_key, group in working.groupby(group_columns, dropna=False, sort=True):
        if len(group_columns) == 1 and not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        probabilities = group[prob_columns].to_numpy(dtype=float)
        group_labels = np.asarray(group["true_label"].tolist(), dtype=object)
        label_positions = results._label_positions(group_labels, probability_label_values)
        row["n_observations"] = int(len(group))
        row["ece"] = results.expected_calibration_error(probabilities, label_positions, n_bins=n_bins)
        rows.append(row)

    return pd.DataFrame(rows)


def _install_scalar_arraylike_guards() -> None:
    results = importlib.import_module("neureptrace.results")
    tables = importlib.import_module("neureptrace.results.tables")

    original_validate_positive_integer = results._validate_positive_integer
    if not getattr(original_validate_positive_integer, _POSITIVE_INTEGER_ARRAYLIKE_ATTR, False):

        def _validate_positive_integer_checked(value: object, *, name: str) -> int:
            if _is_arraylike_scalar_control(value):
                raise ValueError(f"{name} must be a positive integer.")
            return original_validate_positive_integer(value, name=name)

        setattr(_validate_positive_integer_checked, _POSITIVE_INTEGER_ARRAYLIKE_ATTR, True)
        _validate_positive_integer_checked.__wrapped__ = original_validate_positive_integer
        results._validate_positive_integer = _validate_positive_integer_checked

    original_finite_numeric_scalar = tables._finite_numeric_scalar
    if not getattr(original_finite_numeric_scalar, _FINITE_SCALAR_ARRAYLIKE_ATTR, False):

        def _finite_numeric_scalar_checked(value: object, *, name: str) -> float:
            if _is_arraylike_scalar_control(value):
                raise ValueError(f"{name} must be a finite numeric value.")
            return original_finite_numeric_scalar(value, name=name)

        setattr(_finite_numeric_scalar_checked, _FINITE_SCALAR_ARRAYLIKE_ATTR, True)
        _finite_numeric_scalar_checked.__wrapped__ = original_finite_numeric_scalar
        tables._finite_numeric_scalar = _finite_numeric_scalar_checked


def _install_complex_metric_table_guards() -> None:
    results = importlib.import_module("neureptrace.results")
    tables = importlib.import_module("neureptrace.results.tables")
    original = tables.summarize_metric_table
    if getattr(original, _COMPLEX_METRIC_TABLE_ATTR, False):
        results.summarize_metric_table = original
        return

    @wraps(original)
    def summarize_metric_table_checked(
        frame: pd.DataFrame,
        value_column: str,
        group_columns: Sequence[str] | str | None,
        participant_column: str | None = None,
        chance_column: str | None = None,
        scale: float = 1.0,
        *,
        percent_scale: float | None = None,
        percent_prefix: str = "percent",
        chance_percent_column: str | None = None,
        chance_class_columns: Sequence[str] | str | None = None,
        permutation_p_column: str | None = None,
        p_value_thresholds: Sequence[float] = (0.05, 0.01),
        zero_singleton_dispersion: bool = False,
    ) -> pd.DataFrame:
        columns = (
            value_column,
            chance_column,
            permutation_p_column,
            *_column_names(chance_class_columns),
        )
        for column in columns:
            if column is None or column not in frame.columns:
                continue
            rows = _complex_rows(frame[column])
            if rows:
                raise ValueError(
                    f"Metric table column '{column}' must contain real-valued numeric values; "
                    f"complex value row(s): {rows}."
                )
        return original(
            frame,
            value_column,
            group_columns,
            participant_column=participant_column,
            chance_column=chance_column,
            scale=scale,
            percent_scale=percent_scale,
            percent_prefix=percent_prefix,
            chance_percent_column=chance_percent_column,
            chance_class_columns=chance_class_columns,
            permutation_p_column=permutation_p_column,
            p_value_thresholds=p_value_thresholds,
            zero_singleton_dispersion=zero_singleton_dispersion,
        )

    setattr(summarize_metric_table_checked, _COMPLEX_METRIC_TABLE_ATTR, True)
    summarize_metric_table_checked.__wrapped__ = original
    tables.summarize_metric_table = summarize_metric_table_checked
    results.summarize_metric_table = summarize_metric_table_checked


def _install_exact_probability_observation_labels() -> None:
    results = importlib.import_module("neureptrace.results")
    original = results._probability_ece_by_group
    if getattr(original, _EXACT_OBSERVATION_LABEL_ATTR, False):
        return

    setattr(_probability_ece_by_group_exact, _EXACT_OBSERVATION_LABEL_ATTR, True)
    _probability_ece_by_group_exact.__wrapped__ = original
    results._probability_ece_by_group = _probability_ece_by_group_exact


def install() -> None:
    """Install singleton-aware condition alignment and result scalar guards."""

    _install_scalar_arraylike_guards()
    _install_complex_metric_table_guards()
    _install_exact_probability_observation_labels()

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
