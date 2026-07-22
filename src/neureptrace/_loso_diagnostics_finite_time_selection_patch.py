"""Guard LOSO diagnostic time selection and exact integer labels."""

from __future__ import annotations

import importlib
from decimal import Decimal, InvalidOperation
from functools import wraps
from numbers import Integral, Rational, Real
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_loso_diagnostics_finite_time_selection_patch_installed"
_INTEGER_ARRAY_PATCH_MARKER = "_neureptrace_loso_diagnostics_exact_integer_labels_patch_installed"


def _finite_positions(frame: pd.DataFrame, *columns: str) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values)
    return np.flatnonzero(mask)


def _is_missing_scalar(value: object) -> bool:
    """Return whether a scalar is a pandas/NumPy missing value."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _exact_integer(value: object, *, name: str) -> int:
    """Parse one integral label without a lossy binary-float round trip."""

    numeric_error = f"Observation table {name} values must be numeric and non-missing."
    integer_error = f"Observation table {name} values must be integer-valued."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(integer_error)
    if _is_missing_scalar(value):
        raise ValueError(numeric_error)

    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Rational):
        if value.denominator != 1:
            raise ValueError(integer_error)
        return int(value.numerator)
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(integer_error)
        return int(numeric)
    else:
        if isinstance(value, bytes):
            try:
                text = value.decode().strip()
            except UnicodeDecodeError as exc:
                raise ValueError(numeric_error) from exc
        else:
            text = str(value).strip()
        if not text:
            raise ValueError(numeric_error)
        try:
            decimal_value = Decimal(text)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(numeric_error) from exc

    if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
        raise ValueError(integer_error)
    return int(decimal_value)


def _install_best_time_patch(module: Any) -> None:
    original_best_time = module._best_time
    if getattr(original_best_time, _PATCH_MARKER, False):
        return

    @wraps(original_best_time)
    def _best_time(summary: pd.DataFrame, metric: str) -> float:
        if metric not in module.SELECTION_METRICS:
            raise ValueError(f"Unknown selection metric '{metric}'. Available metrics: {', '.join(module.SELECTION_METRICS)}.")
        if summary.empty:
            raise ValueError("Cannot select a best time from an empty summary.")
        if "time" not in summary.columns:
            raise ValueError("Time-course summary must contain 'time'.")
        if metric not in summary.columns:
            raise ValueError(f"Time-course summary must contain '{metric}'.")

        positions = _finite_positions(summary, "time", metric)
        if positions.size == 0:
            raise ValueError(f"Time-course summary must contain at least one finite '{metric}' row with finite time.")

        values = pd.to_numeric(summary[metric], errors="coerce").to_numpy(dtype=float)
        if metric in module.MINIMIZE_METRICS:
            selected_position = positions[int(np.argmin(values[positions]))]
        else:
            selected_position = positions[int(np.argmax(values[positions]))]
        return float(summary.iloc[int(selected_position)]["time"])

    setattr(_best_time, _PATCH_MARKER, True)
    module._best_time = _best_time


def _install_integer_array_patch(module: Any) -> None:
    original_integer_array = module._integer_array
    if getattr(original_integer_array, _INTEGER_ARRAY_PATCH_MARKER, False):
        return

    @wraps(original_integer_array)
    def _integer_array(values: pd.Series, *, name: str) -> np.ndarray:
        parsed = [_exact_integer(value, name=name) for value in values.tolist()]
        limits = np.iinfo(int)
        if any(value < limits.min or value > limits.max for value in parsed):
            raise ValueError(f"Observation table {name} values must fit the platform integer range.")
        return np.asarray(parsed, dtype=int)

    setattr(_integer_array, _INTEGER_ARRAY_PATCH_MARKER, True)
    module._integer_array = _integer_array


def install() -> None:
    """Make LOSO diagnostic time and label handling finite and exact."""

    module = importlib.import_module("neureptrace.loso_observation_diagnostics")
    _install_best_time_patch(module)
    _install_integer_array_patch(module)


__all__ = ["install"]
