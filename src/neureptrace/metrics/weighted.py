"""Weighted probability-metric helpers.

These functions complement the unweighted public metrics in ``neureptrace.metrics``
for aggregation settings where trials, runs, or subjects should not contribute
equally by raw row count.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def validate_sample_weight(sample_weight: Iterable[float] | np.ndarray, n_samples: int) -> np.ndarray:
    """Return validated non-negative per-sample weights.

    Parameters
    ----------
    sample_weight:
        One-dimensional non-negative weights.
    n_samples:
        Expected number of samples.
    """
    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim != 1:
        raise ValueError("sample_weight must have shape (n_samples,)")
    if weights.shape[0] != n_samples:
        raise ValueError("sample_weight and probabilities must contain the same samples")
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be non-negative")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("sample_weight must have positive total weight")
    return weights


def _validate_probability_inputs(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one sample and one class")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (n_samples,)")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must contain the same samples")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.asarray(labels, dtype=int))):
            raise ValueError("labels must contain integer class indices")
        labels = labels.astype(int)
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels must be valid column indices for probabilities")
    return probabilities, labels.astype(int, copy=False)


def weighted_brier_score_multiclass(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weight: Iterable[float] | np.ndarray,
) -> float:
    """Compute a weighted multiclass Brier score using one-hot targets."""
    probabilities, labels = _validate_probability_inputs(probabilities, labels)
    weights = validate_sample_weight(sample_weight, probabilities.shape[0])

    targets = np.zeros_like(probabilities, dtype=float)
    targets[np.arange(labels.shape[0]), labels] = 1.0
    losses = np.sum((probabilities - targets) ** 2, axis=1)
    return float(np.average(losses, weights=weights))


def weighted_negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weight: Iterable[float] | np.ndarray,
    *,
    eps: float = 1e-15,
) -> float:
    """Compute weighted mean categorical negative log-likelihood."""
    probabilities, labels = _validate_probability_inputs(probabilities, labels)
    weights = validate_sample_weight(sample_weight, probabilities.shape[0])
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite value")

    true_probabilities = probabilities[np.arange(labels.shape[0]), labels]
    losses = -np.log(np.clip(true_probabilities, eps, 1.0))
    return float(np.average(losses, weights=weights))


def weighted_top_k_accuracy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weight: Iterable[float] | np.ndarray,
    *,
    k: int = 1,
) -> float:
    """Compute weighted top-k classification accuracy."""
    probabilities, labels = _validate_probability_inputs(probabilities, labels)
    weights = validate_sample_weight(sample_weight, probabilities.shape[0])
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive")
    if k >= probabilities.shape[1]:
        return 1.0

    top_k = np.argpartition(probabilities, kth=probabilities.shape[1] - k, axis=1)[:, -k:]
    correct = np.any(top_k == labels[:, None], axis=1).astype(float)
    return float(np.average(correct, weights=weights))


def weighted_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weight: Iterable[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute weighted top-label expected calibration error."""
    probabilities, labels = _validate_probability_inputs(probabilities, labels)
    weights = validate_sample_weight(sample_weight, probabilities.shape[0])
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    total_weight = float(np.sum(weights))

    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        if not np.any(in_bin):
            continue
        bin_weights = weights[in_bin]
        bin_weight_sum = float(np.sum(bin_weights))
        if bin_weight_sum <= 0.0:
            continue
        bin_accuracy = float(np.average(correct[in_bin].astype(float), weights=bin_weights))
        bin_confidence = float(np.average(confidences[in_bin], weights=bin_weights))
        ece += (bin_weight_sum / total_weight) * abs(bin_accuracy - bin_confidence)
    return float(ece)


__all__ = [
    "validate_sample_weight",
    "weighted_brier_score_multiclass",
    "weighted_expected_calibration_error",
    "weighted_negative_log_likelihood",
    "weighted_top_k_accuracy",
]
