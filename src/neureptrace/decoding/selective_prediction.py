"""Selective prediction utilities for probability traces.

The helpers in this module turn classifier probability rows into predictions plus
an abstention mask.  Fixed thresholds are ordinary post-hoc prediction rules.  A
coverage-calibrated threshold uses the unlabeled target probability distribution
itself and is therefore marked as an unlabeled target-adaptive Category-2 step.

No label vector is accepted by the public prediction API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SELECTIVE_PREDICTION_PROTOCOL = "probability_selective_prediction"
SELECTIVE_PREDICTION_CATEGORY_FIXED = "1_fixed_threshold_prediction_rule"
SELECTIVE_PREDICTION_CATEGORY_COVERAGE = "2_unlabeled_target_adaptive_threshold"
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SelectivePredictionResult:
    """Predictions, retained-row mask, and uncertainty diagnostics."""

    predictions: np.ndarray
    selected_mask: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray
    threshold: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of rows retained for prediction."""

        return float(np.mean(self.selected_mask)) if self.selected_mask.size else 0.0


# pylint: disable-next=too-many-arguments,too-many-locals

def selective_predict(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    confidence_threshold: float | str | None = None,
    max_entropy: float | str | None = None,
    min_margin: float | str | None = None,
    target_coverage: float | str | None = None,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SelectivePredictionResult:
    """Predict with optional abstention from probability rows.

    Parameters
    ----------
    probabilities:
        Row-wise class probabilities or non-negative scores.  Rows are normalized
        defensively before metrics are computed.
    classes:
        Optional class labels for prediction output.  If omitted, dense integer
        class ids are used.
    confidence_threshold:
        Fixed minimum top-class probability required for selection.
    max_entropy:
        Fixed maximum entropy allowed for selection.
    min_margin:
        Fixed minimum gap between the largest and second-largest probability.
    target_coverage:
        Optional desired retained fraction in ``(0, 1]``.  When supplied, the
        confidence threshold is set from the unlabeled probability batch so that
        approximately this coverage is retained.  This is a Category-2 threshold
        selection rule.
    epsilon:
        Numerical floor for row normalization and entropy.

    Returns
    -------
    SelectivePredictionResult
        Predicted labels, selection mask, and uncertainty metrics.
    """

    eps = _positive_float(epsilon, name="epsilon")
    probs = normalize_probability_rows(probabilities, epsilon=eps)
    class_values = _classes(classes, n_classes=probs.shape[1])
    confidence = np.max(probs, axis=1)
    prediction_indices = np.argmax(probs, axis=1)
    predictions = class_values[prediction_indices]
    entropy = probability_entropy(probs, epsilon=eps)
    margin = probability_margin(probs)

    threshold = None if confidence_threshold is None else _unit_interval_float(confidence_threshold, name="confidence_threshold")
    coverage_requested = None if target_coverage is None else _open_unit_interval_float(target_coverage, name="target_coverage", include_one=True)
    if coverage_requested is not None:
        threshold = confidence_threshold_for_coverage(confidence, target_coverage=coverage_requested)

    selected = np.ones(probs.shape[0], dtype=bool)
    if threshold is not None:
        selected &= confidence >= threshold
    if max_entropy is not None:
        selected &= entropy <= _nonnegative_float(max_entropy, name="max_entropy")
    if min_margin is not None:
        selected &= margin >= _unit_interval_float(min_margin, name="min_margin")

    metadata = _metadata(
        n_rows=probs.shape[0],
        n_classes=probs.shape[1],
        selected_count=int(np.count_nonzero(selected)),
        confidence_threshold=threshold,
        max_entropy=max_entropy,
        min_margin=min_margin,
        target_coverage=coverage_requested,
    )
    return SelectivePredictionResult(
        predictions=predictions,
        selected_mask=selected,
        probabilities=probs.astype(np.float32, copy=False),
        confidence=confidence.astype(float, copy=False),
        entropy=entropy.astype(float, copy=False),
        margin=margin.astype(float, copy=False),
        threshold=threshold,
        metadata=metadata,
    )


def normalize_probability_rows(probabilities: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Return finite row-normalized probabilities."""

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must contain at least one row and two classes.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must contain finite non-negative values.")
    eps = _positive_float(epsilon, name="epsilon")
    matrix = np.maximum(matrix, eps)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def probability_entropy(probabilities: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Return row-wise Shannon entropy."""

    probs = normalize_probability_rows(probabilities, epsilon=epsilon)
    return -np.sum(probs * np.log(np.maximum(probs, float(epsilon))), axis=1)


def probability_margin(probabilities: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return top-one minus top-two probability margin per row."""

    probs = normalize_probability_rows(probabilities)
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def confidence_threshold_for_coverage(confidence: Sequence[float] | np.ndarray, *, target_coverage: float | str) -> float:
    """Return the smallest confidence among the top-coverage rows."""

    values = np.asarray(confidence, dtype=float).reshape(-1)
    if values.size < 1:
        raise ValueError("confidence must contain at least one row.")
    if not np.all(np.isfinite(values)):
        raise ValueError("confidence must be finite.")
    coverage = _open_unit_interval_float(target_coverage, name="target_coverage", include_one=True)
    keep = max(1, int(np.ceil(values.size * coverage)))
    ordered = np.sort(values)[::-1]
    return float(ordered[keep - 1])


def _classes(classes: Sequence[Any] | np.ndarray | None, *, n_classes: int) -> np.ndarray:
    if classes is None:
        return np.arange(n_classes)
    values = np.asarray(classes, dtype=object).reshape(-1)
    if values.shape[0] != n_classes:
        raise ValueError(f"classes must contain one label per probability column: {values.shape[0]} != {n_classes}.")
    return values


def _metadata(
    *,
    n_rows: int,
    n_classes: int,
    selected_count: int,
    confidence_threshold: float | None,
    max_entropy: float | str | None,
    min_margin: float | str | None,
    target_coverage: float | None,
) -> dict[str, Any]:
    adaptive = target_coverage is not None
    return {
        "selective_prediction": True,
        "selective_prediction_protocol": SELECTIVE_PREDICTION_PROTOCOL,
        "selective_prediction_protocol_category": SELECTIVE_PREDICTION_CATEGORY_COVERAGE if adaptive else SELECTIVE_PREDICTION_CATEGORY_FIXED,
        "selective_prediction_uses_probabilities": True,
        "selective_prediction_uses_labels": False,
        "selective_prediction_adaptive_threshold": bool(adaptive),
        "selective_prediction_valid_for_fixed_threshold_reporting": not adaptive,
        "selective_prediction_valid_for_unlabeled_target_adaptation": True,
        "selective_prediction_n_rows": int(n_rows),
        "selective_prediction_n_classes": int(n_classes),
        "selective_prediction_selected_count": int(selected_count),
        "selective_prediction_coverage": float(selected_count / max(1, n_rows)),
        "selective_prediction_confidence_threshold": "" if confidence_threshold is None else float(confidence_threshold),
        "selective_prediction_max_entropy": "" if max_entropy is None else float(max_entropy),
        "selective_prediction_min_margin": "" if min_margin is None else float(min_margin),
        "selective_prediction_target_coverage": "" if target_coverage is None else float(target_coverage),
    }


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _open_unit_interval_float(value: float | str, *, name: str, include_one: bool = False) -> float:
    parsed = _float_value(value, name=name)
    upper_ok = parsed <= 1.0 if include_one else parsed < 1.0
    if parsed <= 0.0 or not upper_ok:
        bracket = "(0, 1]" if include_one else "(0, 1)"
        raise ValueError(f"{name} must be in {bracket}.")
    return parsed


def _float_value(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed
