"""Strict source-only confidence-based sample weighting.

This module converts source-row probability estimates into training sample weights.
It is intended for fold-internal stacking or reweighting: the weights are computed
from source probabilities and optional source labels only.  No held-out labels are
part of the API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_CONFIDENCE_WEIGHT_PROTOCOL = "strict_source_only_confidence_weighting"
SOURCE_CONFIDENCE_WEIGHT_CATEGORY = "1_strict_source_only"
WEIGHT_MODES = ("confidence", "correct_confidence", "margin", "entropy")
DEFAULT_MIN_WEIGHT = 0.05
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceConfidenceWeightConfig:
    """Configuration for source-only confidence weighting."""

    mode: str = "confidence"
    min_weight: float = DEFAULT_MIN_WEIGHT
    normalize_weights: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceConfidenceWeightResult:
    """Computed sample weights and provenance metadata."""

    sample_weights: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_source_confidence_weights(
    source_probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_labels: Sequence[int] | np.ndarray | None = None,
    config: SourceConfidenceWeightConfig | Mapping[str, Any] | None = None,
) -> SourceConfidenceWeightResult:
    """Compute source-only sample weights from probability rows.

    Parameters
    ----------
    source_probabilities:
        Source-row class probabilities.
    source_labels:
        Optional source class indices.  Required only for ``correct_confidence``.
    config:
        Weighting options.  A mapping is normalized through
        :func:`source_confidence_weight_config`.
    """

    cfg = source_confidence_weight_config() if config is None else _coerce_config(config)
    probabilities = _probability_matrix(source_probabilities, epsilon=cfg.epsilon)
    labels = None if source_labels is None else _label_indices(source_labels, expected_length=probabilities.shape[0], n_classes=probabilities.shape[1])
    scores = confidence_scores(probabilities, labels=labels, mode=cfg.mode, epsilon=cfg.epsilon)
    weights = np.maximum(scores, cfg.min_weight)
    if cfg.normalize_weights:
        weights = weights / float(np.mean(weights))
    metadata = {
        "source_confidence_weighting": True,
        "source_confidence_weighting_protocol": SOURCE_CONFIDENCE_WEIGHT_PROTOCOL,
        "source_confidence_weighting_protocol_category": SOURCE_CONFIDENCE_WEIGHT_CATEGORY,
        "source_confidence_weighting_mode": cfg.mode,
        "source_confidence_weighting_uses_source_probabilities": True,
        "source_confidence_weighting_uses_source_labels": labels is not None,
        "source_confidence_weighting_uses_heldout_labels": False,
        "source_confidence_weighting_valid_for_strict_source_only": True,
        "source_confidence_weighting_valid_for_benchmark": True,
        "source_confidence_weighting_n_rows": int(probabilities.shape[0]),
        "source_confidence_weighting_n_classes": int(probabilities.shape[1]),
        "source_confidence_weighting_min_weight": float(cfg.min_weight),
        "source_confidence_weighting_normalize_weights": bool(cfg.normalize_weights),
        "source_confidence_weighting_score_min": float(np.min(scores)),
        "source_confidence_weighting_score_max": float(np.max(scores)),
        "source_confidence_weighting_weight_min": float(np.min(weights)),
        "source_confidence_weighting_weight_max": float(np.max(weights)),
    }
    return SourceConfidenceWeightResult(
        sample_weights=weights.astype(np.float32, copy=False),
        scores=scores.astype(np.float32, copy=False),
        metadata=metadata,
    )


def confidence_scores(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    labels: Sequence[int] | np.ndarray | None = None,
    mode: str = "confidence",
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Return source-only confidence scores in [0, 1]."""

    probs = _probability_matrix(probabilities, epsilon=epsilon)
    resolved = normalize_confidence_weight_mode(mode)
    if resolved == "confidence":
        return np.max(probs, axis=1)
    if resolved == "correct_confidence":
        if labels is None:
            raise ValueError("labels are required for correct_confidence mode.")
        label_indices = _label_indices(labels, expected_length=probs.shape[0], n_classes=probs.shape[1])
        return probs[np.arange(probs.shape[0]), label_indices]
    if resolved == "margin":
        sorted_prob = np.sort(probs, axis=1)
        if probs.shape[1] == 1:
            return np.ones(probs.shape[0], dtype=float)
        return sorted_prob[:, -1] - sorted_prob[:, -2]
    if resolved == "entropy":
        entropy = -np.sum(probs * np.log(np.maximum(probs, epsilon)), axis=1)
        max_entropy = np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
        return 1.0 - entropy / max(max_entropy, epsilon)
    raise ValueError(f"Unhandled confidence weighting mode {resolved!r}.")


def source_confidence_weight_config(
    *,
    mode: str | None = "confidence",
    min_weight: float | str = DEFAULT_MIN_WEIGHT,
    normalize_weights: bool | int | str = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceConfidenceWeightConfig:
    """Normalize confidence-weighting options."""

    return SourceConfidenceWeightConfig(
        mode=normalize_confidence_weight_mode(mode),
        min_weight=_unit_interval_float(min_weight, name="min_weight"),
        normalize_weights=_bool_value(normalize_weights, name="normalize_weights"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_confidence_weight_mode(value: str | None) -> str:
    """Normalize confidence-weighting aliases."""

    normalized = "confidence" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "max": "confidence",
        "max_prob": "confidence",
        "true_confidence": "correct_confidence",
        "label_confidence": "correct_confidence",
        "prob_margin": "margin",
        "low_entropy": "entropy",
    }.get(normalized, normalized)
    if normalized not in WEIGHT_MODES:
        raise ValueError(f"Unknown source confidence weighting mode {value!r}.")
    return normalized


def _coerce_config(config: SourceConfidenceWeightConfig | Mapping[str, Any]) -> SourceConfidenceWeightConfig:
    if isinstance(config, SourceConfidenceWeightConfig):
        return config
    return source_confidence_weight_config(**dict(config))


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("source_probabilities must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("source_probabilities must be finite and non-negative.")
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("source_probabilities rows must have positive probability mass.")
    matrix = matrix / row_sums
    matrix = np.maximum(matrix, float(epsilon))
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _label_indices(values: Sequence[int] | np.ndarray, *, expected_length: int, n_classes: int) -> np.ndarray:
    shape_message = "source_labels must be one-dimensional."
    value_message = "source_labels must contain integer class indices."
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(value_message) from exc
    if raw.dtype == np.dtype(bool) or _contains_boolean_label(raw):
        raise ValueError(value_message)
    raw = np.squeeze(raw)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    if raw.ndim != 1:
        raise ValueError(shape_message)
    if raw.shape[0] != expected_length:
        raise ValueError("source_labels must contain one label per probability row.")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(value_message) from exc
    if not np.all(np.isfinite(numeric)) or not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError(value_message)
    labels = numeric.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("source_labels contain class indices outside the probability columns.")
    return labels


def _contains_boolean_label(values: np.ndarray) -> bool:
    if values.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in values.reshape(-1).tolist())


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _bool_value(value: bool | int | str, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")
