"""Accept single string column arguments in observation validation.

The public observation validator accepts optional grouping and stream-column
arguments.  Because strings are sequences, passing a single column name such as
``group_columns="subject"`` used to be interpreted as the characters
``"s"``, ``"u"``, ... and reported as missing columns.  Normalize these API
arguments before the validator checks them.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_observation_schema_string_columns_patch_installed"


def _normalize_column_argument(columns: Sequence[str] | str | None) -> list[str] | None:
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    return list(dict.fromkeys(columns))


def install() -> None:
    """Patch observation validation so one string means one column name."""

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


__all__ = ["install"]
