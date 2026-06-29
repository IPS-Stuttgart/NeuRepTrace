"""Confidence scoring helpers for decoder probability matrices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ConfidenceSelectionResult:
    """Row-wise confidence scores and acceptance mask."""

    confidence: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    predicted_index: np.ndarray
    accepted_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def confidence_scores(probabilities: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return confidence, top-two margin, entropy, and predicted index.

    Rows are normalized internally, so callers may pass non-negative scores whose
    rows have positive mass.
    """

    matrix = _probability_matrix(probabilities)
    # Use a stable descending order so exact probability ties follow NumPy's
    # argmax convention and select the first/lowest class index.
    order = np.argsort(-matrix, axis=1, kind="mergesort")
    top = order[:, 0]
    second = order[:, 1]
    rows = np.arange(matrix.shape[0])
    confidence = matrix[rows, top]
    margin = confidence - matrix[rows, second]
    entropy = -np.sum(matrix * np.log(np.maximum(matrix, 1e-12)), axis=1) / np.log(matrix.shape[1])
    return (
        confidence.astype(np.float32, copy=False),
        margin.astype(np.float32, copy=False),
        entropy.astype(np.float32, copy=False),
        top.astype(int, copy=False),
    )


def select_confident_rows(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    min_confidence: float | str = 0.0,
    min_margin: float | str = 0.0,
    max_entropy: float | str = 1.0,
) -> ConfidenceSelectionResult:
    """Select rows that pass confidence, margin, and entropy thresholds."""

    conf_thr = _unit_interval(min_confidence, name="min_confidence")
    margin_thr = _unit_interval(min_margin, name="min_margin")
    entropy_thr = _unit_interval(max_entropy, name="max_entropy")
    confidence, margin, entropy, predicted = confidence_scores(probabilities)
    accepted = (confidence >= conf_thr) & (margin >= margin_thr) & (entropy <= entropy_thr)
    metadata = {
        "confidence_selection": True,
        "confidence_selection_min_confidence": float(conf_thr),
        "confidence_selection_min_margin": float(margin_thr),
        "confidence_selection_max_entropy": float(entropy_thr),
        "confidence_selection_n_rows": int(accepted.shape[0]),
        "confidence_selection_n_accepted": int(np.count_nonzero(accepted)),
        "confidence_selection_acceptance_rate": float(np.mean(accepted)) if accepted.size else 0.0,
        "confidence_selection_mean_confidence": float(np.mean(confidence)) if confidence.size else 0.0,
        "confidence_selection_mean_margin": float(np.mean(margin)) if margin.size else 0.0,
        "confidence_selection_mean_entropy": float(np.mean(entropy)) if entropy.size else 0.0,
    }
    return ConfidenceSelectionResult(
        confidence=confidence,
        margin=margin,
        entropy=entropy,
        predicted_index=predicted,
        accepted_mask=accepted,
        metadata=metadata,
    )


def accepted_probability_rows(probabilities: Sequence[Sequence[float]] | np.ndarray, *, selection: ConfidenceSelectionResult) -> np.ndarray:
    """Return only rows accepted by a previous confidence selection result."""

    matrix = _probability_matrix(probabilities)
    mask = np.asarray(selection.accepted_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != matrix.shape[0]:
        raise ValueError("selection mask length must match probability rows.")
    return matrix[mask].astype(np.float32, copy=False)


def _contains_boolean_value(values: Any) -> bool:
    array = np.asarray(values, dtype=object)
    return any(isinstance(value, (bool, np.bool_)) for value in array.reshape(-1))


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    if _contains_boolean_value(values):
        raise ValueError("probabilities must contain numeric scores, not booleans.")
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must contain finite non-negative values.")
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _scalar_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        raise ValueError(f"{name} must be a scalar in [0, 1].")
    return value


def _unit_interval(value: float | str | np.ndarray, *, name: str) -> float:
    value = _scalar_value(value, name=name)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be in [0, 1].")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1].") from exc
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed
