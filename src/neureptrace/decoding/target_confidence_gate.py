"""Unlabeled target confidence gating for probability rows.

This module selects confident target predictions from source-model probability
rows without using held-out target labels.  It can use a fixed confidence
threshold or a target-batch quantile/retain-fraction threshold, making it a small
Category-2 post-processing utility for semi-automatic evaluation, rejection, or
pseudo-label pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TARGET_CONFIDENCE_GATE_PROTOCOL = "unlabeled_target_confidence_gate"
TARGET_CONFIDENCE_GATE_CATEGORY = "2_unlabeled_target_adaptive"
SCORE_MODES = ("max_probability", "margin", "normalized_confidence")
THRESHOLD_MODES = ("fixed", "retain_fraction")
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_RETAIN_FRACTION = 0.8
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TargetConfidenceGateConfig:
    """Configuration for unlabeled target confidence gating."""

    score: str = "max_probability"
    threshold_mode: str = "fixed"
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    retain_fraction: float = DEFAULT_RETAIN_FRACTION
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class TargetConfidenceGateResult:
    """Confidence scores, accepted mask, and predictions."""

    probabilities: np.ndarray
    confidence_scores: np.ndarray
    accepted_mask: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rejected_mask(self) -> np.ndarray:
        """Boolean mask for rows rejected by the confidence gate."""

        return ~self.accepted_mask


def gate_target_probabilities_by_confidence(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: TargetConfidenceGateConfig | Mapping[str, Any] | None = None,
) -> TargetConfidenceGateResult:
    """Select confident target probability rows without target labels.

    Parameters
    ----------
    probabilities:
        Target-batch probability rows, usually emitted by a source-trained model.
    classes:
        Optional class labels matching probability columns.  If omitted, integer
        column indices are used.
    config:
        Gating options.  A mapping is normalized through
        :func:`target_confidence_gate_config`.
    """

    cfg = target_confidence_gate_config() if config is None else _coerce_config(config)
    prob = _probability_matrix(probabilities, epsilon=cfg.epsilon)
    class_values = _class_vector(classes, n_classes=prob.shape[1])
    scores = target_confidence_scores(prob, score=cfg.score, epsilon=cfg.epsilon)
    threshold = _threshold(scores, cfg)
    accepted = scores >= threshold
    predictions = class_values[np.argmax(prob, axis=1)]
    metadata = {
        "target_confidence_gate": True,
        "target_confidence_gate_protocol": TARGET_CONFIDENCE_GATE_PROTOCOL,
        "target_confidence_gate_protocol_category": TARGET_CONFIDENCE_GATE_CATEGORY,
        "target_confidence_gate_uses_target_probabilities": True,
        "target_confidence_gate_uses_target_features": False,
        "target_confidence_gate_uses_target_labels": False,
        "target_confidence_gate_valid_for_strict_source_only": False,
        "target_confidence_gate_valid_for_unlabeled_target_adaptation": True,
        "target_confidence_gate_valid_for_benchmark": False,
        "target_confidence_gate_n_rows": int(prob.shape[0]),
        "target_confidence_gate_n_classes": int(prob.shape[1]),
        "target_confidence_gate_n_accepted": int(np.count_nonzero(accepted)),
        "target_confidence_gate_n_rejected": int(np.count_nonzero(~accepted)),
        "target_confidence_gate_score": cfg.score,
        "target_confidence_gate_threshold_mode": cfg.threshold_mode,
        "target_confidence_gate_threshold": float(threshold),
        "target_confidence_gate_confidence_threshold": float(cfg.confidence_threshold),
        "target_confidence_gate_retain_fraction": float(cfg.retain_fraction),
        "target_confidence_gate_epsilon": float(cfg.epsilon),
    }
    return TargetConfidenceGateResult(
        probabilities=prob.astype(np.float32, copy=False),
        confidence_scores=scores.astype(np.float32, copy=False),
        accepted_mask=accepted.astype(bool, copy=False),
        predictions=predictions,
        classes=class_values,
        threshold=float(threshold),
        metadata=metadata,
    )


def target_confidence_gate_config(
    *,
    score: str | None = "max_probability",
    threshold_mode: str | None = "fixed",
    confidence_threshold: float | str = DEFAULT_CONFIDENCE_THRESHOLD,
    retain_fraction: float | str = DEFAULT_RETAIN_FRACTION,
    epsilon: float | str = DEFAULT_EPSILON,
) -> TargetConfidenceGateConfig:
    """Normalize target confidence gate options."""

    return TargetConfidenceGateConfig(
        score=normalize_score_mode(score),
        threshold_mode=normalize_threshold_mode(threshold_mode),
        confidence_threshold=_unit_interval_float(confidence_threshold, name="confidence_threshold"),
        retain_fraction=_open_unit_float(retain_fraction, name="retain_fraction"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_score_mode(value: str | None) -> str:
    """Normalize confidence score aliases."""

    normalized = "max_probability" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "max": "max_probability",
        "probability": "max_probability",
        "top1": "max_probability",
        "top_gap": "margin",
        "top2_margin": "margin",
        "entropy": "normalized_confidence",
        "one_minus_entropy": "normalized_confidence",
    }.get(normalized, normalized)
    if normalized not in SCORE_MODES:
        raise ValueError(f"Unknown confidence score {value!r}. Available values: {', '.join(SCORE_MODES)}.")
    return normalized


def normalize_threshold_mode(value: str | None) -> str:
    """Normalize threshold mode aliases."""

    normalized = "fixed" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"constant": "fixed", "quantile": "retain_fraction", "top_fraction": "retain_fraction"}.get(normalized, normalized)
    if normalized not in THRESHOLD_MODES:
        raise ValueError(f"Unknown threshold_mode {value!r}. Available values: {', '.join(THRESHOLD_MODES)}.")
    return normalized


def target_confidence_scores(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    score: str = "max_probability",
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Compute label-free confidence scores from probability rows."""

    prob = _probability_matrix(probabilities, epsilon=epsilon)
    mode = normalize_score_mode(score)
    if mode == "max_probability":
        return np.max(prob, axis=1)
    if mode == "margin":
        if prob.shape[1] < 2:
            raise ValueError("margin confidence requires at least two classes.")
        sorted_prob = np.sort(prob, axis=1)
        return sorted_prob[:, -1] - sorted_prob[:, -2]
    if mode == "normalized_confidence":
        entropy = -np.sum(prob * np.log(np.maximum(prob, epsilon)), axis=1)
        max_entropy = np.log(prob.shape[1])
        return 1.0 - entropy / max(max_entropy, epsilon)
    raise ValueError(f"Unhandled score mode {mode!r}.")


def _threshold(scores: np.ndarray, cfg: TargetConfidenceGateConfig) -> float:
    if cfg.threshold_mode == "fixed":
        return float(cfg.confidence_threshold)
    if cfg.threshold_mode == "retain_fraction":
        quantile = 1.0 - float(cfg.retain_fraction)
        return float(np.quantile(scores, quantile))
    raise ValueError(f"Unhandled threshold_mode {cfg.threshold_mode!r}.")


def _coerce_config(config: TargetConfidenceGateConfig | Mapping[str, Any]) -> TargetConfidenceGateConfig:
    if isinstance(config, TargetConfidenceGateConfig):
        return config
    return target_confidence_gate_config(**dict(config))


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float) -> np.ndarray:
    materialized = _materialize_nested_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError("probabilities must contain numeric probabilities, not boolean values.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    row_maxima = np.max(matrix, axis=1, keepdims=True)
    if np.any(row_maxima <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    floored = np.maximum(matrix, float(epsilon))
    scaled = floored / np.max(floored, axis=1, keepdims=True)
    return scaled / np.sum(scaled, axis=1, keepdims=True)


def _class_vector(values: Sequence[Any] | np.ndarray | None, *, n_classes: int) -> np.ndarray:
    if values is None:
        return np.arange(n_classes, dtype=int)
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != n_classes:
        raise ValueError(f"classes must contain one value per probability column: {vector.shape[0]} != {n_classes}.")
    return vector


def _materialize_nested_iterables(values: Any) -> Any:
    if isinstance(values, np.ndarray) or _is_scalar_like(values):
        return values
    if isinstance(values, Mapping):
        return values
    if isinstance(values, Iterable):
        return [list(row) if _is_row_iterable(row) else row for row in values]
    return values


def _is_row_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, np.ndarray) and not _is_scalar_like(value) and not isinstance(value, Mapping)


def _is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, bytes))


def _contains_boolean_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype != object:
            return False
        return any(_contains_boolean_value(item) for item in value.flat)
    if _is_scalar_like(value):
        return False
    if isinstance(value, Mapping):
        return any(_contains_boolean_value(item) for pair in value.items() for item in pair)
    if isinstance(value, Iterable):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _open_unit_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return parsed
