"""Normalize reaction-time scalar conversion and association arithmetic."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation

import numpy as np

_PATCH_MARKER = "_neureptrace_reaction_time_trial_value_type_patch_installed"
_CLEAN_ID_PATCH_MARKER = "_neureptrace_reaction_time_clean_id_patch_installed"
_FLOAT_PATCH_MARKER = "_neureptrace_reaction_time_float_missing_patch_installed"
_VALUES_PATCH_MARKER = "_neureptrace_reaction_time_values_missing_patch_installed"
_ASSOCIATION_MEAN_PATCH_MARKER = "_neureptrace_reaction_time_association_mean_patch_installed"
_WITHIN_CENTER_PATCH_MARKER = "_neureptrace_reaction_time_within_center_patch_installed"


def _safe_repr(value: object) -> str:
    """Return a diagnostic representation without masking conversion errors."""

    try:
        return repr(value)
    except Exception:  # pragma: no cover - only exotic objects fail during repr
        return f"<{type(value).__name__}>"


def _trial_error(value: object) -> ValueError:
    return ValueError(f"trial values must be finite integers, got {_safe_repr(value)}.")


def _clean_id(value: object) -> str:
    """Normalize integer-like identifiers without lossy float conversion."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if number.is_finite() and number == number.to_integral_value():
        return str(int(number))
    return text


def _to_float(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _to_int(value: object) -> int:
    """Parse an integer-valued trial key exactly, including large integers."""

    try:
        text = "" if value is None else str(value).strip()
        number = Decimal(text)
    except (TypeError, ValueError, OverflowError, InvalidOperation) as exc:
        raise _trial_error(value) from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise _trial_error(value)
    return int(number)


def _materialize_value_sequence(values: object) -> object:
    """Preserve scalar values while expanding one-pass iterables once."""

    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, (str, bytes, Mapping)):
        return values
    if isinstance(values, Iterable):
        return list(values)
    return values


def _numeric_values(values: object) -> list[float]:
    array = np.asarray(_materialize_value_sequence(values), dtype=object).ravel()
    return [_to_float(value) for value in array]


def _stable_mean(values: np.ndarray) -> float:
    """Return the mean of finite values without overflowing the reduction."""

    if values.size == 0:
        return np.nan
    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0.0:
        return 0.0
    return float(np.mean(values / magnitude) * magnitude)


def _empty_association(
    scope: str,
    participant: str,
    metric: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, object]:
    return {
        "scope": scope,
        "participant": participant,
        "metric": metric,
        "n_trials": int(x_values.size),
        "metric_mean": _stable_mean(x_values),
        "reaction_time_mean": _stable_mean(y_values),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "slope_reaction_time_per_unit": np.nan,
        "intercept_reaction_time": np.nan,
    }


def _within_participant_centered_rows(
    module,
    grouped_rows,
    metric: str,
    reaction_time_column: str,
) -> list[dict[str, object]]:
    centered_rows: list[dict[str, object]] = []
    for participant_rows in grouped_rows.values():
        x_values, y_values = module._finite_metric_arrays(participant_rows, metric, reaction_time_column)
        x_mean = _stable_mean(x_values)
        y_mean = _stable_mean(y_values)
        for x_value, y_value in zip(x_values, y_values):
            centered_rows.append(
                {
                    "participant": "",
                    metric: x_value - x_mean,
                    reaction_time_column: y_value - y_mean,
                }
            )
    return centered_rows


def install() -> None:
    """Ensure invalid scalars and extreme finite means are handled explicitly."""

    module = importlib.import_module("neureptrace.behavior.reaction_time")

    original_clean_id = module._clean_id
    if not getattr(original_clean_id, _CLEAN_ID_PATCH_MARKER, False):
        setattr(_clean_id, _CLEAN_ID_PATCH_MARKER, True)
        module._clean_id = _clean_id

    original_to_int = module._to_int
    if not getattr(original_to_int, _PATCH_MARKER, False):
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

    original_empty_association = module._empty_association
    if not getattr(original_empty_association, _ASSOCIATION_MEAN_PATCH_MARKER, False):
        setattr(_empty_association, _ASSOCIATION_MEAN_PATCH_MARKER, True)
        module._empty_association = _empty_association

    original_within_participant_centered_rows = module._within_participant_centered_rows
    if not getattr(original_within_participant_centered_rows, _WITHIN_CENTER_PATCH_MARKER, False):

        def within_participant_centered_rows(grouped_rows, metric, reaction_time_column):
            return _within_participant_centered_rows(module, grouped_rows, metric, reaction_time_column)

        setattr(within_participant_centered_rows, _WITHIN_CENTER_PATCH_MARKER, True)
        module._within_participant_centered_rows = within_participant_centered_rows


__all__ = ["install"]
