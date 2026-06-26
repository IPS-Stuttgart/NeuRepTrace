"""Confidence filtering utilities for probability traces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CONFIDENCE_FILTER_PROTOCOL = "posthoc_probability_confidence_filter"


@dataclass(frozen=True, slots=True)
class ConfidenceFilterResult:
    """Top-class confidence, margin, and accepted-row mask."""

    confidence: np.ndarray
    margin: np.ndarray
    predicted_index: np.ndarray
    accepted_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def confidence_filter(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    min_confidence: float | str = 0.0,
    min_margin: float | str = 0.0,
    max_entropy: float | str | None = None,
    normalize_entropy: bool = True,
) -> ConfidenceFilterResult:
    """Return a row mask from confidence, margin, and optional entropy rules.

    The function consumes probability rows only.  It is useful for selective
    decoding, pseudo-label candidate selection, and reporting coverage at fixed
    confidence thresholds.
    """

    matrix = _probability_matrix(probabilities)
    confidence_threshold = _unit_interval(min_confidence, name="min_confidence")
    margin_threshold = _unit_interval(min_margin, name="min_margin")
    entropy_threshold = None if max_entropy in {None, "", "none", "None"} else _nonnegative_float(max_entropy, name="max_entropy")
    order = np.argsort(matrix, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    row_index = np.arange(matrix.shape[0])
    confidence = matrix[row_index, top]
    margin = confidence - matrix[row_index, second]
    accepted = (confidence >= confidence_threshold) & (margin >= margin_threshold)
    entropy = probability_entropy(matrix, normalize=normalize_entropy)
    if entropy_threshold is not None:
        accepted &= entropy <= entropy_threshold
    metadata = {
        "confidence_filter": True,
        "confidence_filter_protocol": CONFIDENCE_FILTER_PROTOCOL,
        "confidence_filter_min_confidence": float(confidence_threshold),
        "confidence_filter_min_margin": float(margin_threshold),
        "confidence_filter_max_entropy": "" if entropy_threshold is None else float(entropy_threshold),
        "confidence_filter_entropy_normalized": bool(normalize_entropy),
        "confidence_filter_n_rows": int(matrix.shape[0]),
        "confidence_filter_n_classes": int(matrix.shape[1]),
        "confidence_filter_n_accepted": int(np.count_nonzero(accepted)),
        "confidence_filter_acceptance_rate": float(np.mean(accepted)),
        "confidence_filter_mean_confidence": float(np.mean(confidence)),
        "confidence_filter_mean_margin": float(np.mean(margin)),
        "confidence_filter_mean_entropy": float(np.mean(entropy)),
    }
    return ConfidenceFilterResult(
        confidence=confidence.astype(np.float32, copy=False),
        margin=margin.astype(np.float32, copy=False),
        predicted_index=top.astype(int, copy=False),
        accepted_mask=accepted.astype(bool, copy=False),
        metadata=metadata,
    )


def probability_entropy(probabilities: Sequence[Sequence[float]] | np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Return entropy for each probability row."""

    matrix = _probability_matrix(probabilities)
    entropy = -np.sum(matrix * np.log(np.maximum(matrix, 1e-12)), axis=1)
    if normalize:
        entropy = entropy / np.log(matrix.shape[1])
    return entropy.astype(np.float32, copy=False)


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must contain finite non-negative values.")
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _unit_interval(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed
