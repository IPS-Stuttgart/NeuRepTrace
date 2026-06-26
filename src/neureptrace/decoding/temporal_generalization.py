from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFeatureWindow:
    """Feature matrix and labels for one train or test time window."""

    center: float
    features: Any
    labels: Sequence
    start: float | None = None
    stop: float | None = None
    metadata: Mapping[str, object] | None = None


def compute_temporal_generalization_matrix(
    train_windows: Sequence[TemporalFeatureWindow],
    test_windows: Sequence[TemporalFeatureWindow],
    *,
    fit_model: Callable[[TemporalFeatureWindow], Any],
    predict_labels: Callable[[Any, TemporalFeatureWindow], Sequence],
    chance_accuracy: float | None = None,
    metadata: Mapping[str, object] | None = None,
    model_metadata: Callable[[Any], Mapping[str, object]] | None = None,
    center_decimals: int = 10,
) -> pd.DataFrame:
    """Compute a train-time by test-time temporal-generalization score table.

    NeuRepTrace owns the dataset-independent orchestration and scoring: train one
    model per train window, evaluate it on every test window, and emit a compact
    figure-independent table. Dataset-specific projects remain responsible for
    loading data and constructing ``TemporalFeatureWindow`` objects.
    """

    center_decimals = _validate_center_decimals(center_decimals)
    fixed_chance_accuracy = _validate_chance_accuracy(chance_accuracy)

    if not train_windows:
        raise ValueError("Need at least one train window.")
    if not test_windows:
        raise ValueError("Need at least one test window.")

    base_metadata = dict(metadata or {})
    rows: list[dict[str, object]] = []
    for train_window in sorted(train_windows, key=lambda window: _center_key(window.center, center_decimals)):
        train_labels = _validate_window_labels(train_window, role="train")
        model = fit_model(train_window)
        fitted_metadata = dict(model_metadata(model) if model_metadata is not None else {})
        for test_window in sorted(test_windows, key=lambda window: _center_key(window.center, center_decimals)):
            labels = _validate_window_labels(test_window, role="test")
            predictions = _label_vector(predict_labels(model, test_window), role="predicted labels")
            if len(predictions) != len(labels):
                raise ValueError(
                    "Predicted label count must match test label count "
                    f"for test window {test_window.center}: {len(predictions)} != {len(labels)}."
                )
            accuracy = float(np.mean(predictions == labels)) if len(labels) else np.nan
            chance = _chance_accuracy(fixed_chance_accuracy, labels)
            rows.append(
                {
                    **base_metadata,
                    **_window_metadata("train", train_window),
                    **_window_metadata("test", test_window),
                    "is_diagonal": _center_key(train_window.center, center_decimals) == _center_key(test_window.center, center_decimals),
                    "accuracy": accuracy,
                    "percent": 100.0 * accuracy if np.isfinite(accuracy) else np.nan,
                    "chance_accuracy": chance,
                    "chance_percent": 100.0 * chance if np.isfinite(chance) else np.nan,
                    "above_chance": bool(np.isfinite(accuracy) and np.isfinite(chance) and accuracy > chance),
                    "n_train_trials": len(train_labels),
                    "n_validation_trials": len(labels),
                    "n_train_classes": _unique_label_count(train_labels),
                    "n_validation_classes": _unique_label_count(labels),
                    **fitted_metadata,
                }
            )
    return pd.DataFrame(rows)


def summarize_temporal_generalization_matrix(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
    accuracy_column: str = "accuracy",
    chance_column: str | None = "chance_accuracy",
) -> pd.DataFrame:
    """Summarize temporal-generalization rows across participants or repeats."""

    if frame.empty:
        return pd.DataFrame()
    required_columns = set(group_columns) | {accuracy_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")

    grouped = frame.groupby(list(group_columns), sort=True, dropna=False) if group_columns else [((), frame)]
    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        values = pd.to_numeric(group[accuracy_column], errors="coerce").dropna().to_numpy(dtype=float)
        mean_value = float(np.mean(values)) if len(values) else np.nan
        median_value = float(np.median(values)) if len(values) else np.nan
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sem_value = float(std_value / np.sqrt(len(values))) if len(values) > 1 else 0.0
        row: dict[str, object] = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "n_rows": int(len(group)),
                f"{accuracy_column}_mean": mean_value,
                f"{accuracy_column}_median": median_value,
                f"{accuracy_column}_std": std_value,
                f"{accuracy_column}_sem": sem_value,
            }
        )
        row.update(
            {
                "percent_mean": 100.0 * mean_value if np.isfinite(mean_value) else np.nan,
                "percent_median": 100.0 * median_value if np.isfinite(median_value) else np.nan,
                "percent_std": 100.0 * std_value if np.isfinite(std_value) else np.nan,
                "percent_sem": 100.0 * sem_value if np.isfinite(sem_value) else np.nan,
            }
        )
        if chance_column is not None and chance_column in group.columns:
            chance_values = pd.to_numeric(group[chance_column], errors="coerce").dropna()
            chance = float(chance_values.iloc[0]) if not chance_values.empty else np.nan
            row["chance_accuracy"] = chance
            row["chance_percent"] = 100.0 * chance if np.isfinite(chance) else np.nan
            row["above_chance_count"] = int((values > chance).sum()) if np.isfinite(chance) else 0
        if "is_diagonal" in group.columns:
            diagonal_values = set(group["is_diagonal"].astype(bool))
            row["is_diagonal"] = bool(diagonal_values == {True})
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_window_labels(window: TemporalFeatureWindow, *, role: str) -> np.ndarray:
    labels = _label_vector(window.labels, role=f"{role} window labels")
    if len(labels) == 0:
        raise ValueError(f"{role} window labels must not be empty.")
    return labels


def _label_vector(labels: Sequence, *, role: str) -> np.ndarray:
    if isinstance(labels, (str, bytes)):
        raise ValueError(f"{role} must be one-dimensional.")
    if isinstance(labels, np.ndarray):
        label_array = np.asarray(labels)
        if label_array.ndim == 1:
            values = label_array.astype(object, copy=False).tolist()
        elif label_array.ndim == 2 and 1 in label_array.shape:
            values = label_array.reshape(-1).astype(object, copy=False).tolist()
        elif label_array.ndim == 2 and label_array.dtype == object:
            values = [tuple(_normalize_atomic_label(item) for item in row.tolist()) for row in label_array]
        else:
            raise ValueError(f"{role} must be one-dimensional.")
    else:
        try:
            values = list(labels)
        except TypeError as exc:
            raise ValueError(f"{role} must be one-dimensional.") from exc
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = _normalize_atomic_label(value)
    return result


def _normalize_atomic_label(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_atomic_label(value.item())
        return tuple(_normalize_atomic_label(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_normalize_atomic_label(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_atomic_label(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unique_label_count(labels: Sequence) -> int:
    unique_labels: list[Any] = []
    for label in labels:
        if not any(_label_equal(label, existing) for existing in unique_labels):
            unique_labels.append(label)
    return len(unique_labels)


def _label_equal(left: Any, right: Any) -> bool:
    if _is_nan_scalar(left) and _is_nan_scalar(right):
        return True
    try:
        return bool(left == right)
    except ValueError:
        return False


def _is_nan_scalar(value: Any) -> bool:
    try:
        return bool(np.isscalar(value) and np.isnan(value))
    except TypeError:
        return False


def _validate_center_decimals(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("center_decimals must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("center_decimals must be an integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError("center_decimals must be an integer.")
    return int(numeric)


def _validate_chance_accuracy(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("chance_accuracy must be a finite probability in [0, 1].")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("chance_accuracy must be a finite probability in [0, 1].") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError("chance_accuracy must be a finite probability in [0, 1].")
    return numeric


def _center_key(center: float, decimals: int) -> float:
    return round(float(center), decimals)


def _chance_accuracy(chance_accuracy: float | None, labels: np.ndarray) -> float:
    if chance_accuracy is not None:
        return float(chance_accuracy)
    n_classes = _unique_label_count(labels)
    return 1.0 / n_classes if n_classes else np.nan


def _window_metadata(prefix: str, window: TemporalFeatureWindow) -> dict[str, object]:
    metadata = dict(window.metadata or {})
    return {
        f"{prefix}_window_center_s": float(window.center),
        f"{prefix}_window_start_s": float(window.start) if window.start is not None else np.nan,
        f"{prefix}_window_stop_s": float(window.stop) if window.stop is not None else np.nan,
        **metadata,
    }
