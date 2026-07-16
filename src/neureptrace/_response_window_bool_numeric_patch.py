"""Reject boolean numerics and preserve exact signed probability-observation labels.

Pandas and NumPy treat booleans as numeric during coercion. They also commonly
coerce integer-like values through ``float64``, which cannot distinguish adjacent
integer class identifiers above ``2**53``. Probability-observation workflows must
reject booleans without corrupting otherwise valid signed integer labels.
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


def _integer_probability_suffix(suffix: str) -> int | None:
    """Parse a decimal integer suffix, including an optional sign."""

    text = str(suffix)
    digits = text[1:] if text[:1] in {"+", "-"} else text
    if not digits or not digits.isdigit():
        return None
    return int(text)


def _probability_sort_key(column: str) -> tuple[int, str]:
    suffix = column.removeprefix("prob_class_")
    label = _integer_probability_suffix(suffix)
    return (label, suffix) if label is not None else (10_000, suffix)


def _numeric_probability_labels(columns: Sequence[str]) -> tuple[int, ...] | None:
    labels = tuple(
        _integer_probability_suffix(column.removeprefix("prob_class_"))
        for column in columns
    )
    if any(label is None for label in labels):
        return None
    return tuple(int(label) for label in labels if label is not None)


def _duplicate_probability_labels(labels: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    duplicates: list[int] = []
    for label in labels:
        if label in seen and label not in duplicates:
            duplicates.append(label)
        seen.add(label)
    return duplicates


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(
        (column for column in frame.columns if column.startswith("prob_class_")),
        key=_probability_sort_key,
    )
    if not columns:
        raise ValueError("Observation CSVs must contain probability columns named 'prob_class_*'.")
    labels = _numeric_probability_labels(columns)
    if labels is not None:
        duplicates = _duplicate_probability_labels(labels)
        if duplicates:
            raise ValueError(
                "prob_class_* columns must map to unique class labels; "
                f"duplicate label(s): {duplicates}."
            )
    return columns


def _label_values(prob_columns: Sequence[str]) -> tuple[int, ...]:
    labels = _numeric_probability_labels(prob_columns)
    if labels is None:
        return tuple(range(len(prob_columns)))
    return labels


def _class_names(frame: pd.DataFrame, prob_columns: Sequence[str]) -> list[str]:
    names: list[str] = []
    for index, column in enumerate(prob_columns):
        suffix = column.removeprefix("prob_class_")
        class_column = f"class_{suffix}"
        if class_column in frame.columns and frame[class_column].notna().any():
            names.append(str(frame[class_column].dropna().iloc[0]))
        else:
            names.append(suffix if _integer_probability_suffix(suffix) is not None else str(index))
    return names


def install() -> None:
    """Patch probability validation and exact signed label conversion."""

    observations = importlib.import_module("neureptrace.observations")
    temporal_model = importlib.import_module("neureptrace.temporal_model")
    temporal_smoothing = importlib.import_module("neureptrace.temporal_smoothing")
    response_window_ensemble = importlib.import_module("neureptrace.response_window_ensemble")

    if (
        getattr(observations, _PATCH_MARKER, False)
        and getattr(temporal_model, _PATCH_MARKER, False)
        and getattr(temporal_smoothing, _PATCH_MARKER, False)
        and getattr(response_window_ensemble, _PATCH_MARKER, False)
    ):
        return

    if not getattr(observations, _PATCH_MARKER, False):
        observations._probability_sort_key = _probability_sort_key
        observations._numeric_probability_labels = _numeric_probability_labels
        setattr(observations, _PATCH_MARKER, True)

    if not getattr(temporal_model, _PATCH_MARKER, False):
        original_validate_probability_matrix = temporal_model._validate_probability_matrix

        @wraps(original_validate_probability_matrix)
        def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
            if _contains_boolean(probabilities):
                raise ValueError("Probability observations must be numeric probabilities, not booleans.")
            return original_validate_probability_matrix(probabilities)

        temporal_model._validate_probability_matrix = _validate_probability_matrix
        temporal_model.probability_columns = _probability_columns
        temporal_model._class_names = _class_names
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

        temporal_smoothing.probability_columns = temporal_model.probability_columns
        temporal_smoothing._class_names = temporal_model._class_names
        temporal_smoothing._label_values = _label_values
        temporal_smoothing._numeric_label_values = _numeric_label_values
        setattr(temporal_smoothing, _PATCH_MARKER, True)

    if not getattr(response_window_ensemble, _PATCH_MARKER, False):
        original_integer_label_values = response_window_ensemble._integer_label_values

        @wraps(original_integer_label_values)
        def _integer_label_values(
            values: Sequence[object] | np.ndarray | pd.Series,
            *,
            n_classes: int | None = None,
        ) -> np.ndarray:
            labels = _exact_integer_labels(values, label_name="Response-window true_label")
            if n_classes is not None and bool(((labels < 0) | (labels >= int(n_classes))).any()):
                raise ValueError("Response-window true_label values must index prob_class_* columns.")
            return labels

        response_window_ensemble.probability_columns = temporal_model.probability_columns
        response_window_ensemble._label_values = _label_values
        response_window_ensemble._class_names = _class_names
        response_window_ensemble._integer_label_values = _integer_label_values
        setattr(response_window_ensemble, _PATCH_MARKER, True)


__all__ = ["install"]
