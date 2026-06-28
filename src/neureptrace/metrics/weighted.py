"""Weighted probability-metric helpers.

These functions complement the unweighted public metrics in ``neureptrace.metrics``
for aggregation settings where trials, runs, or subjects should not contribute
equally by raw row count.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _weights_contain_boolean(weights: np.ndarray) -> bool:
    if np.issubdtype(weights.dtype, np.bool_):
        return True
    if weights.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in weights.ravel())
    return False


def _sample_weight_array(sample_weight: Iterable[float] | np.ndarray) -> np.ndarray:
    """Return sample weights as an object array without dropping one-pass iterables."""

    try:
        if isinstance(sample_weight, np.ndarray) or isinstance(sample_weight, (str, bytes)):
            return np.asarray(sample_weight, dtype=object)
        return np.asarray(list(sample_weight), dtype=object)
    except TypeError:
        return np.asarray(sample_weight, dtype=object)
    except ValueError as exc:
        raise ValueError("sample_weight must have shape (n_samples,)") from exc


def validate_sample_weight(sample_weight: Iterable[float] | np.ndarray, n_samples: int) -> np.ndarray:
    """Return validated non-negative per-sample weights.

    Parameters
    ----------
    sample_weight:
        One-dimensional non-negative weights.
    n_samples:
        Expected number of samples.
    """
    try:
        raw_weights = _sample_weight_array(sample_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_weight must have shape (n_samples,)") from exc
    if _weights_contain_boolean(raw_weights):
        raise ValueError("sample_weight must contain numeric weights, not boolean values")
    try:
        weights = raw_weights.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_weight must contain numeric weights") from exc
    if weights.ndim == 2 and weights.shape[1] == 1:
        weights = weights.reshape(-1)
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


def _validate_probability_inputs(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one sample and one class")
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
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
    labels = _coerce_label_indices(labels)
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels must be valid column indices for probabilities")
    return probabilities, labels


def _validate_n_bins(n_bins: int) -> int:
    if isinstance(n_bins, (bool, np.bool_)):
        raise ValueError("n_bins must be a positive integer")
    try:
        numeric = float(n_bins)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_bins must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric < 1.0 or numeric % 1.0 != 0.0:
        raise ValueError("n_bins must be a positive integer")
    return int(numeric)


def _validate_k(k: int) -> int:
    if isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be a positive integer")
    try:
        numeric = float(k)
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric < 1.0 or numeric % 1.0 != 0.0:
        raise ValueError("k must be a positive integer")
    return int(numeric)


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
    if not np.isfinite(eps) or eps <= 0.0 or eps >= 1.0:
        raise ValueError("eps must be finite and in the open interval (0, 1)")

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
    """Compute weighted top-k classification accuracy.

    Probability ties are resolved deterministically by class-index order. This
    keeps the selected top-k set size equal to ``k`` and prevents uniform or
    exactly tied probability rows from being counted as correct for every class.
    """
    probabilities, labels = _validate_probability_inputs(probabilities, labels)
    weights = validate_sample_weight(sample_weight, probabilities.shape[0])
    k = _validate_k(k)
    if k >= probabilities.shape[1]:
        return 1.0

    top_k = np.argsort(-probabilities, axis=1, kind="mergesort")[:, :k]
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
    n_bins = _validate_n_bins(n_bins)

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


def weighted_reliability_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weight: Iterable[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Summarize weighted top-label reliability bins for calibration plots.

    The returned rows keep the unweighted ``reliability_bins`` schema and add
    ``sample_weight`` plus ``sample_weight_fraction`` so downstream reports can
    display both raw-bin occupancy and the effective contribution of each bin.
    """
    probabilities, labels = _validate_probability_inputs(probabilities, labels)
    weights = validate_sample_weight(sample_weight, probabilities.shape[0])
    n_bins = _validate_n_bins(n_bins)

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    total_weight = float(np.sum(weights))

    rows: list[dict[str, float | int]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right == 1.0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        n_samples = int(np.sum(in_bin))
        bin_weight_sum = float(np.sum(weights[in_bin])) if n_samples else 0.0
        if n_samples and bin_weight_sum > 0.0:
            bin_weights = weights[in_bin]
            accuracy = float(np.average(correct[in_bin].astype(float), weights=bin_weights))
            confidence = float(np.average(confidences[in_bin], weights=bin_weights))
            gap = accuracy - confidence
        else:
            accuracy = float("nan")
            confidence = float("nan")
            gap = float("nan")
        rows.append(
            {
                "bin": bin_index,
                "bin_left": float(left),
                "bin_right": float(right),
                "n_samples": n_samples,
                "sample_weight": bin_weight_sum,
                "sample_weight_fraction": bin_weight_sum / total_weight,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": gap,
            }
        )
    return rows


__all__ = [
    "validate_sample_weight",
    "weighted_brier_score_multiclass",
    "weighted_expected_calibration_error",
    "weighted_negative_log_likelihood",
    "weighted_reliability_bins",
    "weighted_top_k_accuracy",
]
