"""Normalize reaction-time scalar conversion errors."""

from __future__ import annotations

import importlib

import numpy as np

_PATCH_MARKER = "_neureptrace_reaction_time_trial_value_type_patch_installed"
_FLOAT_PATCH_MARKER = "_neureptrace_reaction_time_float_missing_patch_installed"
_VALUES_PATCH_MARKER = "_neureptrace_reaction_time_values_missing_patch_installed"


def _trial_error(value: object) -> ValueError:
    return ValueError(f"trial values must be finite integers, got {value!r}.")


def _to_float(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _numeric_values(values: object) -> list[float]:
    array = np.asarray(values, dtype=object).ravel()
    return [_to_float(value) for value in array]


def install() -> None:
    """Ensure invalid trial objects and missing RT scalars are handled explicitly."""

    module = importlib.import_module("neureptrace.behavior.reaction_time")
    original_to_int = module._to_int
    if not getattr(original_to_int, _PATCH_MARKER, False):

        def _to_int(value: object) -> int:
            try:
                text = "" if value is None else str(value).strip()
                number = float(text)
            except (TypeError, ValueError) as exc:
                raise _trial_error(value) from exc
            if not np.isfinite(number) or not number.is_integer():
                raise _trial_error(value)
            return int(number)

        setattr(_to_int, _PATCH_MARKER, True)
        module._to_int = _to_int

    original_to_float = module._to_float
    if not getattr(original_to_float, _FLOAT_PATCH_MARKER, False):
        setattr(_to_float, _FLOAT_PATCH_MARKER, True)
        module._to_float = _to_float

    original_reaction_time_rows_from_values = module.reaction_time_rows_from_values
    if not getattr(original_reaction_time_rows_from_values, _VALUES_PATCH_MARKER, False):

        def reaction_time_rows_from_values(
            values,
            *,
            participant=None,
            dataset="main",
            reaction_time_scale=1.0,
        ):
            return original_reaction_time_rows_from_values(
                _numeric_values(values),
                participant=participant,
                dataset=dataset,
                reaction_time_scale=reaction_time_scale,
            )

        setattr(reaction_time_rows_from_values, _VALUES_PATCH_MARKER, True)
        module.reaction_time_rows_from_values = reaction_time_rows_from_values


__all__ = ["install"]
