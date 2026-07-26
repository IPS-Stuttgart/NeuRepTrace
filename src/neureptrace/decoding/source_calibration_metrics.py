"""Strict source-only probability calibration metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_CALIBRATION_METRICS_PROTOCOL = "strict_source_only_probability_calibration_metrics"
SOURCE_CALIBRATION_METRICS_CATEGORY = "1_strict_source_only"


@dataclass(frozen=True, slots=True)
class SourceCalibrationMetricsResult:
    """Calibration metrics and provenance metadata."""

    nll: float
    brier: float
    ece: float
    accuracy: float
    n_bins: int
    metadata: dict[str, Any] = field(default_factory=dict)


def source_calibration_metrics(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int | str = 10,
    epsilon: float | str = 1e-12,
) -> SourceCalibrationMetricsResult:
    """Compute source-only probability calibration metrics.

    The function uses probability rows and integer source-label indices only.  It
    has no access to held-out target labels.
    """

    eps = _positive_probability_floor(epsilon)
    probs = _probability_matrix(probabilities, epsilon=eps)
    label_index = _label_indices(labels, n_rows=probs.shape[0], n_classes=probs.shape[1])
    bins = _positive_int(n_bins, name="n_bins")
    nll = float(-np.mean(np.log(np.maximum(probs[np.arange(probs.shape[0]), label_index], eps))))
    targets = np.zeros_like(probs)
    targets[np.arange(probs.shape[0]), label_index] = 1.0
    brier = float(np.mean(np.sum((probs - targets) ** 2, axis=1)))
    predictions = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    accuracy = float(np.mean(predictions == label_index))
    ece = _expected_calibration_error(confidence, predictions == label_index, n_bins=bins)
    metadata = {
        "source_calibration_metrics": True,
        "source_calibration_metrics_protocol": SOURCE_CALIBRATION_METRICS_PROTOCOL,
        "source_calibration_metrics_protocol_category": SOURCE_CALIBRATION_METRICS_CATEGORY,
        "source_calibration_metrics_uses_source_probabilities": True,
        "source_calibration_metrics_uses_source_labels": True,
        "source_calibration_metrics_uses_heldout_labels": False,
        "source_calibration_metrics_valid_for_strict_source_only": True,
        "source_calibration_metrics_valid_for_benchmark": True,
        "source_calibration_metrics_n_rows": int(probs.shape[0]),
        "source_calibration_metrics_n_classes": int(probs.shape[1]),
        "source_calibration_metrics_n_bins": int(bins),
        "source_calibration_metrics_epsilon": float(eps),
    }
    return SourceCalibrationMetricsResult(nll=nll, brier=brier, ece=ece, accuracy=accuracy, n_bins=bins, metadata=metadata)


def _expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, *, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = confidence.shape[0]
    value = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            value += float(np.count_nonzero(mask)) / float(total) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(value)


def _materialize_numeric_input(value: object) -> object:
    """Materialize nested one-pass iterables before validating numeric values."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_numeric_input(value.tolist())
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__array__"):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_numeric_input(item) for item in value]


def _contains_boolean_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.ravel(order="C"))
        return False
    if hasattr(value, "__array__"):
        try:
            return _contains_boolean_value(np.asarray(value, dtype=object))
        except (TypeError, ValueError):
            return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Iterable):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _contains_complex_value(value: object) -> bool:
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex_value(item) for item in value.ravel(order="C"))
        return False
    if hasattr(value, "__array__"):
        try:
            return _contains_complex_value(np.asarray(value, dtype=object))
        except (TypeError, ValueError):
            return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Iterable):
        return any(_contains_complex_value(item) for item in value)
    return False


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float) -> np.ndarray:
    materialized = _materialize_numeric_input(values)
    if _contains_boolean_value(materialized):
        raise ValueError("probabilities must contain numeric probability values, not boolean flags.")
    if _contains_complex_value(materialized):
        raise ValueError("probabilities must contain real-valued probability values, not complex values.")
    try:
        matrix = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probabilities must contain numeric values.") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must contain finite non-negative values.")
    row_maxima = np.max(matrix, axis=1, keepdims=True)
    if np.any(row_maxima == 0.0):
        raise ValueError("probability rows must have positive mass.")
    scaled = matrix / row_maxima
    scaled_row_sums = np.sum(scaled, axis=1, keepdims=True)
    if np.any(row_maxima <= epsilon / scaled_row_sums):
        raise ValueError("probability rows must have positive mass.")
    return scaled / scaled_row_sums


def _label_indices(values: Sequence[int] | np.ndarray, *, n_rows: int, n_classes: int) -> np.ndarray:
    materialized = _materialize_numeric_input(values)
    if _contains_boolean_value(materialized):
        raise ValueError("labels must contain integer class indices, not boolean flags.")
    if _contains_complex_value(materialized):
        raise ValueError("labels must contain real integer class indices, not complex values.")
    array = np.asarray(materialized)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1 or array.shape[0] != n_rows:
        raise ValueError("labels must contain one value per probability row.")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("labels must contain integer class indices.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("labels must contain integer class indices.")
    indices = numeric.astype(int, copy=False)
    if np.any(indices < 0) or np.any(indices >= n_classes):
        raise ValueError("labels contain class indices outside the probability width.")
    return indices


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_probability_floor(value: float | str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed >= 1.0:
        raise ValueError("epsilon must be positive and smaller than one.")
    return parsed
