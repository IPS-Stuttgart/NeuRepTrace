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


def validate_probability_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    require_normalized: bool = True,
    normalization_atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate and coerce probability-matrix inputs used by scoring metrics.

    Parameters
    ----------
    probabilities:
        Array-like object with shape ``(n_samples, n_classes)``.
    labels:
        Optional integer class labels of shape ``(n_samples,)``.
    require_normalized:
        If true, each probability row must sum to one within
        ``normalization_atol``.
    normalization_atol:
        Absolute tolerance for row-sum checks.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one sample and one class")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(probabilities < -normalization_atol):
        raise ValueError("probabilities must be non-negative")

    row_sums = probabilities.sum(axis=1)
    if require_normalized and not np.allclose(row_sums, 1.0, atol=normalization_atol, rtol=0.0):
        raise ValueError("probability rows must sum to one")

    if labels is None:
        return probabilities, None

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must have shape (n_samples,)")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must contain the same samples")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.asarray(labels, dtype=int))):
            raise ValueError("labels must contain integer class indices")
        labels = labels.astype(int)
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels must be valid column indices for probabilities")
    return probabilities, labels.astype(int, copy=False)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute top-label expected calibration error.

    Parameters
    ----------
    probabilities:
        Array of shape ``(n_samples, n_classes)`` with predicted class probabilities.
    labels:
        Integer class labels of shape ``(n_samples,)``.
    n_bins:
        Number of equally spaced confidence bins.
    """
    probabilities, labels = validate_probability_inputs(probabilities, labels)
    assert labels is not None
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

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


def reliability_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Summarize top-label reliability bins for calibration plots."""
    probabilities, labels = validate_probability_inputs(probabilities, labels)
    assert labels is not None
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

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
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite value")

    true_probabilities = probabilities[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(true_probabilities, eps, 1.0))))


def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int = 1) -> float:
    """Compute top-k classification accuracy from probability rows."""
    probabilities, labels = validate_probability_inputs(probabilities, labels)
    assert labels is not None
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive")
    if k >= probabilities.shape[1]:
        return 1.0

    top_k = np.argpartition(probabilities, kth=probabilities.shape[1] - k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_k == labels[:, None], axis=1)))
