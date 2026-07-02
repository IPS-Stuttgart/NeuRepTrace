"""Patch observation-schema column arguments and temporal sequence keys.

The public observation validator accepts optional grouping and stream-column
arguments.  Because strings are sequences, passing a single column name such as
``group_columns="subject"`` used to be interpreted as the characters
``"s"``, ``"u"``, ... and reported as missing columns.  Normalize these API
arguments before the validator checks them.

The temporal profile should also use the same provenance-aware sequence identity
as the temporal model reader.  Reused ``sequence_id``/``sample_index`` values in
different source files, sessions, or runs must not be concatenated during schema
validation.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_observation_schema_string_columns_patch_installed"


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


def install() -> None:
    """Patch observation validation string handling and temporal keys."""

    observation_schema = importlib.import_module("neureptrace.observation_schema")

    original_validate = observation_schema.validate_probability_observations
    if not getattr(original_validate, _PATCH_MARKER, False):

        @wraps(original_validate)
        def validate_probability_observations(*args: Any, **kwargs: Any):
            if "group_columns" in kwargs:
                kwargs["group_columns"] = _normalize_column_argument(kwargs["group_columns"])
            if "stream_columns" in kwargs:
                kwargs["stream_columns"] = _normalize_column_argument(kwargs["stream_columns"])
            return original_validate(*args, **kwargs)

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


__all__ = ["install"]
