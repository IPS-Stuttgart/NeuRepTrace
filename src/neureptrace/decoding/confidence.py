"""Confidence scoring helpers for decoder probability matrices."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when finite nonzero values survive the conversion."""

    array = np.asarray(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return array
    if np.any((array != 0.0) & (compact == 0.0)):
        return array
    return compact


def _confidence_components(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return full-precision confidence components for threshold decisions."""

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
    return confidence, margin, entropy, top.astype(int, copy=False)


def confidence_scores(probabilities: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return confidence, top-two margin, entropy, and predicted index.

    Rows are normalized internally, so callers may pass non-negative scores whose
    rows have positive mass.
    """

    confidence, margin, entropy, predicted = _confidence_components(probabilities)
    return (
        _compact_float32(confidence),
        _compact_float32(margin),
        _compact_float32(entropy),
        predicted,
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
    confidence, margin, entropy, predicted = _confidence_components(probabilities)
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
    mask = _selection_mask(selection.accepted_mask)
    if mask.shape[0] != matrix.shape[0]:
        raise ValueError("selection mask length must match probability rows.")
    return _compact_float32(matrix[mask])


def _selection_mask(values: Any) -> np.ndarray:
    """Return a one-dimensional Boolean selection mask without truthiness coercion."""

    materialized = _materialize_nested_iterables(values)
    try:
        mask = np.asarray(materialized)
    except (TypeError, ValueError) as exc:
        raise ValueError("selection mask must contain only boolean values.") from exc
    if mask.ndim != 1:
        raise ValueError("selection mask must be one-dimensional.")
    if np.issubdtype(mask.dtype, np.bool_):
        return mask.astype(bool, copy=False)
    if mask.dtype == object and all(isinstance(value, (bool, np.bool_)) for value in mask):
        return mask.astype(bool, copy=False)
    raise ValueError("selection mask must contain only boolean values.")


def _materialize_nested_iterables(values: Any) -> Any:
    """Materialize nested one-pass iterables before validation and coercion."""

    if isinstance(values, np.ndarray):
        if values.dtype != object:
            return values
        materialized = [_materialize_nested_iterables(value) for value in values.ravel(order="C")]
        return np.asarray(materialized, dtype=object).reshape(values.shape)
    if isinstance(values, (str, bytes)):
        return values
    if not isinstance(values, Iterable):
        return values
    return [_materialize_nested_iterables(value) for value in values]


def _contains_boolean_value(values: Any) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return True
        if values.dtype == object:
            return any(_contains_boolean_value(value) for value in values.reshape(-1))
        return False
    if isinstance(values, (str, bytes)):
        return False
    if isinstance(values, Iterable):
        return any(_contains_boolean_value(value) for value in values)
    return False


def _contains_complex_value(values: Any) -> bool:
    if isinstance(values, (complex, np.complexfloating)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.complexfloating):
            return True
        if values.dtype == object:
            return any(_contains_complex_value(value) for value in values.reshape(-1))
        return False
    if hasattr(values, "__array__"):
        try:
            array = np.asarray(values)
        except (TypeError, ValueError):
            return False
        return _contains_complex_value(array)
    if isinstance(values, (str, bytes)):
        return False
    if isinstance(values, Iterable):
        return any(_contains_complex_value(value) for value in values)
    return False


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    materialized = _materialize_nested_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError("probabilities must contain numeric scores, not booleans.")
    if _contains_complex_value(materialized):
        raise ValueError("probabilities must contain real-valued scores, not complex values.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must contain finite non-negative values.")
    row_maxima = np.max(matrix, axis=1, keepdims=True)
    if np.any(row_maxima <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    scaled = matrix / row_maxima
    return scaled / np.sum(scaled, axis=1, keepdims=True)


def _scalar_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar in [0, 1].")
        return value.item()
    return value


def _unit_interval(value: float | str | np.ndarray, *, name: str) -> float:
    value = _scalar_value(value, name=name)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be in [0, 1].")
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f"{name} must be a real scalar in [0, 1].")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1].") from exc
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed
