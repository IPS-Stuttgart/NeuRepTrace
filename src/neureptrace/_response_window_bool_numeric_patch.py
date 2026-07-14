"""Reject boolean numerics and preserve exact probability-observation labels.

Pandas and NumPy treat booleans as numeric during coercion. They also commonly
coerce integer-like values through ``float64``, which cannot distinguish adjacent
integer class identifiers above ``2**53``. Probability-observation workflows must
reject booleans without corrupting otherwise valid integer labels.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_response_window_bool_numeric_patch_installed"
_MAX_EXACT_FLOAT_INTEGER = 2**53
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN_DECIMAL = Decimal(_INT64_MIN)
_INT64_MAX_DECIMAL = Decimal(_INT64_MAX)


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _contains_boolean(values: object) -> bool:
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return bool(array.size)
    if array.dtype == object:
        object_array = np.asarray(values, dtype=object)
        return any(_is_bool_scalar(value) for value in object_array.ravel())
    return False


def _exact_integer_label(value: Any, *, label_name: str) -> int:
    """Return one signed-64-bit label without a lossy float round-trip."""

    if _is_bool_scalar(value):
        raise ValueError(
            f"{label_name} values must be numeric integer labels, not booleans; "
            f"invalid values: [{value!r}]"
        )

    if isinstance(value, (int, np.integer)):
        integer = int(value)
        if integer < _INT64_MIN or integer > _INT64_MAX:
            raise ValueError(f"{label_name} values must fit signed 64-bit integers.")
        return integer

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{label_name} values must be numeric and non-missing.")
        if not numeric.is_integer():
            raise ValueError(f"{label_name} values must be integer-valued.")
        if abs(numeric) > _MAX_EXACT_FLOAT_INTEGER:
            raise ValueError(
                f"{label_name} values must be exact integer labels; values above 2**53 "
                "must be supplied as integers or decimal strings."
            )
        return int(numeric)

    text = str(value).strip()
    if not text:
        raise ValueError(f"{label_name} values must be numeric and non-missing.")
    try:
        numeric = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label_name} values must be numeric and non-missing.") from exc
    if not numeric.is_finite():
        raise ValueError(f"{label_name} values must be numeric and non-missing.")
    integral = numeric.to_integral_value()
    if numeric != integral:
        raise ValueError(f"{label_name} values must be integer-valued.")
    if integral < _INT64_MIN_DECIMAL or integral > _INT64_MAX_DECIMAL:
        raise ValueError(f"{label_name} values must fit signed 64-bit integers.")
    return int(integral)


def _exact_integer_labels(
    values: Sequence[object] | np.ndarray | pd.Series,
    *,
    label_name: str,
) -> np.ndarray:
    return np.asarray(
        [_exact_integer_label(value, label_name=label_name) for value in pd.Series(values).tolist()],
        dtype=np.int64,
    )


def install() -> None:
    """Patch probability validation and exact label conversion."""

    temporal_model = importlib.import_module("neureptrace.temporal_model")
    temporal_smoothing = importlib.import_module("neureptrace.temporal_smoothing")
    response_window_ensemble = importlib.import_module("neureptrace.response_window_ensemble")

    if (
        getattr(temporal_model, _PATCH_MARKER, False)
        and getattr(temporal_smoothing, _PATCH_MARKER, False)
        and getattr(response_window_ensemble, _PATCH_MARKER, False)
    ):
        return

    if not getattr(temporal_model, _PATCH_MARKER, False):
        original_validate_probability_matrix = temporal_model._validate_probability_matrix

        @wraps(original_validate_probability_matrix)
        def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
            if _contains_boolean(probabilities):
                raise ValueError("Probability observations must be numeric probabilities, not booleans.")
            return original_validate_probability_matrix(probabilities)

        temporal_model._validate_probability_matrix = _validate_probability_matrix
        setattr(temporal_model, _PATCH_MARKER, True)

    if not getattr(temporal_smoothing, _PATCH_MARKER, False):
        original_numeric_label_values = temporal_smoothing._numeric_label_values

        @wraps(original_numeric_label_values)
        def _numeric_label_values(frame: pd.DataFrame, label_values: tuple[int, ...]) -> np.ndarray:
            if "true_label" not in frame.columns:
                raise ValueError("Temporal smoothing metrics require a true_label column.")
            labels = _exact_integer_labels(frame["true_label"], label_name="true_label")
            label_set = set(label_values)
            missing = sorted(set(int(label) for label in labels if int(label) not in label_set))
            if missing:
                raise ValueError(
                    f"true_label values must index prob_class_* labels {list(label_values)}; "
                    f"missing labels: {missing[:5]}."
                )
            return labels

        temporal_smoothing._numeric_label_values = _numeric_label_values
        setattr(temporal_smoothing, _PATCH_MARKER, True)

    if not getattr(response_window_ensemble, _PATCH_MARKER, False):
        original_integer_label_values = response_window_ensemble._integer_label_values

        @wraps(original_integer_label_values)
        def _integer_label_values(values: Sequence[object] | np.ndarray | pd.Series, *, n_classes: int | None = None) -> np.ndarray:
            labels = _exact_integer_labels(values, label_name="Response-window true_label")
            if n_classes is not None and bool(((labels < 0) | (labels >= int(n_classes))).any()):
                raise ValueError("Response-window true_label values must index prob_class_* columns.")
            return labels

        response_window_ensemble._integer_label_values = _integer_label_values
        setattr(response_window_ensemble, _PATCH_MARKER, True)


__all__ = ["install"]
