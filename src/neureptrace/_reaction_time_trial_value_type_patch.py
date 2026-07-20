"""Normalize reaction-time scalar conversion and association arithmetic."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation, localcontext

import numpy as np

_PATCH_MARKER = "_neureptrace_reaction_time_trial_value_type_patch_installed"
_CLEAN_ID_PATCH_MARKER = "_neureptrace_reaction_time_clean_id_patch_installed"
_FLOAT_PATCH_MARKER = "_neureptrace_reaction_time_float_missing_patch_installed"
_VALUES_PATCH_MARKER = "_neureptrace_reaction_time_values_missing_patch_installed"
_ASSOCIATION_MEAN_PATCH_MARKER = "_neureptrace_reaction_time_association_mean_patch_installed"
_WITHIN_CENTER_PATCH_MARKER = "_neureptrace_reaction_time_within_center_patch_installed"
_ASSOCIATION_SCALE_PATCH_MARKER = "_neureptrace_reaction_time_association_scale_patch_installed"


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


def _scaled_values(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize a non-constant finite vector without changing association statistics."""

    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0.0:
        return values.copy(), 0.0
    return values / magnitude, magnitude


def _has_variation(values: np.ndarray) -> bool:
    """Check exact variation without subtracting extreme finite values."""

    return bool(values.size and np.any(values != values[0]))


def _rescale_ratio(value: float, numerator: float, denominator: float = 1.0) -> float:
    """Return ``value * numerator / denominator`` without binary64 intermediates."""

    if value == 0.0 or numerator == 0.0:
        return 0.0
    with localcontext() as context:
        context.prec = 80
        scaled = (
            Decimal.from_float(float(value))
            * Decimal.from_float(float(numerator))
            / Decimal.from_float(float(denominator))
        )
        maximum = Decimal.from_float(np.finfo(float).max)
    if scaled > maximum:
        return np.finfo(float).max
    if scaled < -maximum:
        return -np.finfo(float).max
    return float(scaled)


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


def _association_row(
    module,
    scope: str,
    participant: str,
    metric: str,
    rows,
    *,
    reaction_time_column: str,
    min_trials: int,
) -> dict[str, object]:
    """Compute one association after independent overflow-safe axis scaling."""

    x_values, y_values = module._finite_metric_arrays(rows, metric, reaction_time_column)
    result = module._empty_association(scope, participant, metric, x_values, y_values)
    if x_values.size < min_trials or not _has_variation(x_values) or not _has_variation(y_values):
        return result

    from scipy import stats  # pylint: disable=import-outside-toplevel

    scaled_x, x_scale = _scaled_values(x_values)
    scaled_y, y_scale = _scaled_values(y_values)
    pearson = stats.pearsonr(scaled_x, scaled_y)
    regression = stats.linregress(scaled_x, scaled_y)
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "slope_reaction_time_per_unit": _rescale_ratio(float(regression.slope), y_scale, x_scale),
            "intercept_reaction_time": _rescale_ratio(float(regression.intercept), y_scale),
        }
    )
    return result


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
    """Ensure invalid scalars and extreme finite association values are handled explicitly."""

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

    original_association_row = module._association_row
    if not getattr(original_association_row, _ASSOCIATION_SCALE_PATCH_MARKER, False):

        def association_row(
            scope,
            participant,
            metric,
            rows,
            *,
            reaction_time_column,
            min_trials,
        ):
            return _association_row(
                module,
                scope,
                participant,
                metric,
                rows,
                reaction_time_column=reaction_time_column,
                min_trials=min_trials,
            )

        setattr(association_row, _ASSOCIATION_SCALE_PATCH_MARKER, True)
        module._association_row = association_row

    original_within_participant_centered_rows = module._within_participant_centered_rows
    if not getattr(original_within_participant_centered_rows, _WITHIN_CENTER_PATCH_MARKER, False):

        def within_participant_centered_rows(grouped_rows, metric, reaction_time_column):
            return _within_participant_centered_rows(module, grouped_rows, metric, reaction_time_column)

        setattr(within_participant_centered_rows, _WITHIN_CENTER_PATCH_MARKER, True)
        module._within_participant_centered_rows = within_participant_centered_rows


__all__ = ["install"]
