from __future__ import annotations

import numpy as np

from neureptrace.metrics.confusion import confusion_category_enrichment, confusion_category_matrix, confusion_counts, confusion_pair_summary, per_class_accuracy
from neureptrace.metrics.prepost import compare_prepost_windows, summarize_window_metric
from neureptrace.metrics.ranking import rank_class_scores
from neureptrace.metrics.weighted import (
    validate_sample_weight,
    weighted_brier_score_multiclass,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_reliability_bins,
    weighted_top_k_accuracy,
)

__all__ = [
    "brier_score_multiclass",
    "compare_prepost_windows",
    "confusion_category_enrichment",
    "confusion_category_matrix",
    "confusion_counts",
    "confusion_pair_summary",
    "expected_calibration_error",
    "negative_log_likelihood",
    "per_class_accuracy",
    "rank_class_scores",
    "reliability_bins",
    "summarize_window_metric",
    "top_k_accuracy",
    "validate_probability_inputs",
    "validate_sample_weight",
    "weighted_brier_score_multiclass",
    "weighted_expected_calibration_error",
    "weighted_negative_log_likelihood",
    "weighted_reliability_bins",
    "weighted_top_k_accuracy",
]


def _validate_non_negative_finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative finite value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite value") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value")
    return numeric


def _validate_positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite value") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return numeric


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric < 1.0 or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _labels_contain_boolean(labels: np.ndarray) -> bool:
    if np.issubdtype(labels.dtype, np.bool_):
        return True
    if labels.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in labels.ravel())
    return False


def _coerce_label_indices(labels: np.ndarray) -> np.ndarray:
    if _labels_contain_boolean(labels):
        raise ValueError("labels must contain integer class indices")
    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(int, copy=False)

    try:
        numeric_labels = np.asarray(labels, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain integer class indices") from exc
    if not np.all(np.isfinite(numeric_labels)):
        raise ValueError("labels must contain finite integer class indices")
    if not np.all(numeric_labels == np.floor(numeric_labels)):
        raise ValueError("labels must contain integer class indices")
    return numeric_labels.astype(int, copy=False)


def validate_probability_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    require_normalized: bool = True,
    normalization_atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate and coerce probability-matrix inputs used by scoring metrics."""
    normalization_atol = _validate_non_negative_finite_float(normalization_atol, "normalization_atol")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one sample and one class")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(probabilities < -normalization_atol):
        raise ValueError("probabilities must be non-negative")
    if np.any(probabilities < 0.0):
        probabilities = np.maximum(probabilities, 0.0)

    row_sums = probabilities.sum(axis=1)
    if require_normalized:
        if not np.allclose(row_sums, 1.0, atol=normalization_atol, rtol=0.0):
            raise ValueError("probability rows must sum to one")
        probabilities = probabilities / row_sums[:, None]

    if labels is None:
        return probabilities, None

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must have shape (n_samples,)")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must contain the same samples")
    labels = _coerce_label_indices(labels)
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels must be valid column indices for probabilities")
    return probabilities, labels


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    """Compute top-label expected calibration error."""
    probabilities, labels = validate_probability_inputs(probabilities, labels)
    assert labels is not None
    n_bins = _validate_positive_integer(n_bins, "n_bins")

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        if not np.any(in_bin):
            continue
        bin_weight = np.mean(in_bin)
        bin_accuracy = np.mean(correct[in_bin])
        bin_confidence = np.mean(confidences[in_bin])
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(ece)


def reliability_bins(probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> list[dict[str, float | int]]:
    """Summarize top-label reliability bins for calibration plots."""
    probabilities, labels = validate_probability_inputs(probabilities, labels)
    assert labels is not None
    n_bins = _validate_positive_integer(n_bins, "n_bins")

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels

    rows: list[dict[str, float | int]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right == 1.0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        n_samples = int(np.sum(in_bin))
        if n_samples:
            accuracy = float(np.mean(correct[in_bin]))
            confidence = float(np.mean(confidences[in_bin]))
        else:
            accuracy = float("nan")
            confidence = float("nan")
        rows.append(
            {
                "bin": bin_index,
                "bin_left": float(left),
                "bin_right": float(right),
                "n_samples": n_samples,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": accuracy - confidence if n_samples else float("nan"),
            }
        )
    return rows


def brier_score_multiclass(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute multiclass Brier score using one-hot targets."""
    probabilities, labels = validate_probability_inputs(probabilities, labels)
    assert labels is not None

    targets = np.zeros_like(probabilities, dtype=float)
    targets[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray, *, eps: float = 1e-15) -> float:
    """Compute mean categorical negative log-likelihood from probabilities."""
    probabilities, labels = validate_probability_inputs(probabilities, labels)
    assert labels is not None
    eps = _validate_positive_finite_float(eps, "eps")

    true_probabilities = probabilities[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(true_probabilities, eps, 1.0))))


def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int = 1) -> float:
    """Compute top-k classification accuracy from probability rows."""
    probabilities, labels = validate_probability_inputs(probabilities, labels)
    assert labels is not None
    k = _validate_positive_integer(k, "k")
    if k >= probabilities.shape[1]:
        return 1.0

    kth_scores = np.partition(probabilities, kth=probabilities.shape[1] - k, axis=1)[:, -k]
    true_scores = probabilities[np.arange(labels.shape[0]), labels]
    return float(np.mean(true_scores >= kth_scores))
