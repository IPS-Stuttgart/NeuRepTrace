"""Patch observation-schema column arguments, labels, row positions, and temporal keys.

The public observation validator accepts optional grouping and stream-column
arguments. Because strings are sequences, passing a single column name such as
``group_columns="subject"`` used to be interpreted as the characters
``"s"``, ``"u"``, ... and reported as missing columns. Normalize these API
arguments before the validator checks them.

Probability-column discovery must also tolerate ordinary pandas DataFrames with
mixed column-label types. Non-string metadata labels are unrelated to the
``prob_class_*`` schema and should be ignored instead of causing an incidental
``AttributeError`` when ``startswith`` is evaluated.

Validation diagnostics expose integer row numbers, but pandas indexes may use
arbitrary labels. Reset an internal copy to a zero-based RangeIndex so invalid
rows are reported by position instead of crashing while coercing string or
other non-integer index labels.

The temporal profile should also use the same provenance-aware sequence identity
as the temporal model reader. Reused ``sequence_id``/``sample_index`` values in
different source files, sessions, or runs must not be concatenated during schema
validation. Missing or blank sequence identifiers are rejected because grouping
them with ``dropna=False`` would otherwise create an artificial sequence.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_observation_schema_string_columns_patch_installed"
_MISSING_IDENTIFIER_TOKENS = frozenset({"", "<na>", "nan", "nat", "none", "null"})


_SEQUENCE_ID_KEY_CANDIDATES = (
    "source_path",
    "source_file",
    "subject",
    "session",
    "run",
    "fold",
    "sequence_id",
)
_SAMPLE_INDEX_KEY_CANDIDATES = (
    "source_path",
    "source_file",
    "subject",
    "session",
    "run",
    "fold",
    "sample_index",
)


def _normalize_column_argument(columns: Sequence[str] | str | None) -> list[str] | None:
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    return list(dict.fromkeys(columns))


def _temporal_sequence_key_columns(frame: Any) -> list[str]:
    if "sequence_id" in frame.columns:
        return [column for column in _SEQUENCE_ID_KEY_CANDIDATES if column in frame.columns]
    if "sample_index" in frame.columns:
        return [column for column in _SAMPLE_INDEX_KEY_CANDIDATES if column in frame.columns]
    return []


def _temporal_identifier_column(frame: Any) -> str | None:
    if "sequence_id" in frame.columns:
        return "sequence_id"
    if "sample_index" in frame.columns:
        return "sample_index"
    return None


def _missing_identifier_mask(frame: Any, column: str) -> Any:
    values = frame[column]
    missing = values.isna()
    try:
        tokens = values.astype("string").str.strip().str.lower()
    except (AttributeError, TypeError, ValueError):
        return missing
    return missing | tokens.isin(_MISSING_IDENTIFIER_TOKENS).fillna(False)


def _append_temporal_identifier_issues(report: Any, frame: Any, observation_schema: Any) -> Any:
    if report.profile != "temporal-model":
        return report
    column = _temporal_identifier_column(frame)
    if column is None:
        return report
    missing = _missing_identifier_mask(frame, column)
    if not bool(missing.any()):
        return report

    issues = list(report.issues)
    for row_index, value in frame.loc[missing, column].head(20).items():
        issues.append(
            observation_schema.ObservationValidationIssue(
                "error",
                "missing_sequence_identifier_value",
                f"Temporal-model observations must not contain missing or blank values in '{column}'.",
                column=column,
                row=int(row_index),
                value=value,
            )
        )
    if int(missing.sum()) > 20:
        issues.append(
            observation_schema.ObservationValidationIssue(
                "error",
                "missing_sequence_identifier_value_truncated",
                f"Column '{column}' contains {int(missing.sum())} missing or blank temporal sequence identifiers; first 20 are listed.",
                column=column,
            )
        )
    return observation_schema.ObservationValidationReport(
        profile=report.profile,
        n_rows=report.n_rows,
        probability_columns=report.probability_columns,
        issues=tuple(issues),
    )


def _reject_missing_temporal_identifiers(frame: Any) -> None:
    column = _temporal_identifier_column(frame)
    if column is None:
        return
    missing = _missing_identifier_mask(frame, column)
    if bool(missing.any()):
        bad_rows = frame.index[missing].tolist()[:5]
        raise ValueError(
            f"Temporal sequence identifier column '{column}' contains missing or blank values at row(s) {bad_rows}. "
            "Fill every sequence identifier before temporal modeling."
        )


def install() -> None:
    """Patch observation validation column handling, row positions, and temporal keys."""

    observation_schema = importlib.import_module("neureptrace.observation_schema")

    original_probability_columns = observation_schema.probability_columns
    if not getattr(original_probability_columns, _PATCH_MARKER, False):

        @wraps(original_probability_columns)
        def probability_columns(frame: Any) -> list[str]:
            string_positions = [index for index, column in enumerate(frame.columns) if isinstance(column, str)]
            if len(string_positions) == len(frame.columns):
                return original_probability_columns(frame)
            return original_probability_columns(frame.iloc[:, string_positions])

        setattr(probability_columns, _PATCH_MARKER, True)
        observation_schema.probability_columns = probability_columns

    original_validate = observation_schema.validate_probability_observations
    if not getattr(original_validate, _PATCH_MARKER, False):

        @wraps(original_validate)
        def validate_probability_observations(*args: Any, **kwargs: Any):
            if "group_columns" in kwargs:
                kwargs["group_columns"] = _normalize_column_argument(kwargs["group_columns"])
            if "stream_columns" in kwargs:
                kwargs["stream_columns"] = _normalize_column_argument(kwargs["stream_columns"])

            frame = args[0] if args else kwargs.get("frame")
            validation_frame = None
            if frame is not None:
                validation_frame = frame.reset_index(drop=True)
                if args:
                    args = (validation_frame, *args[1:])
                else:
                    kwargs["frame"] = validation_frame

            report = original_validate(*args, **kwargs)
            if validation_frame is None:
                return report
            return _append_temporal_identifier_issues(report, validation_frame, observation_schema)

        setattr(validate_probability_observations, _PATCH_MARKER, True)
        observation_schema.validate_probability_observations = validate_probability_observations

    original_group_validator = observation_schema._validate_group_columns
    if not getattr(original_group_validator, _PATCH_MARKER, False):

        @wraps(original_group_validator)
        def _validate_group_columns(frame: Any, group_columns: Sequence[str] | str | None, issues: list[Any]) -> None:
            return original_group_validator(frame, _normalize_column_argument(group_columns), issues)

        setattr(_validate_group_columns, _PATCH_MARKER, True)
        observation_schema._validate_group_columns = _validate_group_columns

    original_stimulus_validator = observation_schema._validate_stimulus_profile
    if not getattr(original_stimulus_validator, _PATCH_MARKER, False):

        @wraps(original_stimulus_validator)
        def _validate_stimulus_profile(frame: Any, stream_columns: Sequence[str] | str | None, issues: list[Any]) -> None:
            return original_stimulus_validator(frame, _normalize_column_argument(stream_columns), issues)

        setattr(_validate_stimulus_profile, _PATCH_MARKER, True)
        observation_schema._validate_stimulus_profile = _validate_stimulus_profile

    original_sequence_key_columns = observation_schema._sequence_key_columns
    if not getattr(original_sequence_key_columns, _PATCH_MARKER, False):
        setattr(_temporal_sequence_key_columns, _PATCH_MARKER, True)
        observation_schema._sequence_key_columns = _temporal_sequence_key_columns

    temporal_model = importlib.import_module("neureptrace.temporal_model")
    original_reader = temporal_model.read_probability_observations
    if not getattr(original_reader, _PATCH_MARKER, False):

        @wraps(original_reader)
        def read_probability_observations(*args: Any, **kwargs: Any):
            frame = original_reader(*args, **kwargs)
            _reject_missing_temporal_identifiers(frame)
            return frame

        setattr(read_probability_observations, _PATCH_MARKER, True)
        temporal_model.read_probability_observations = read_probability_observations


__all__ = ["install"]
