"""Unlabeled target confidence weighting for Category-2 adaptation.

This module converts target probability rows from a source-trained model into
pseudo-labels, confidence scores, sample weights, and optional keep masks.  It is
intended as a small building block for pseudo-label self-training, target
selection, and reporting.  The public API intentionally has no target-label
argument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TARGET_CONFIDENCE_PROTOCOL = "unlabeled_target_confidence_weighting"
TARGET_CONFIDENCE_CATEGORY = "2_unlabeled_target_adaptive"
WEIGHTING_MODES = ("confidence", "margin", "entropy", "mask")
DEFAULT_CONFIDENCE_THRESHOLD = 0.0
DEFAULT_MIN_KEEP = 1
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TargetConfidenceConfig:
    """Configuration for unlabeled target confidence weighting."""

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    min_keep: int = DEFAULT_MIN_KEEP
    weighting: str = "confidence"
    normalize_weights: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class TargetConfidenceResult:
    """Pseudo-labels, confidence scores, weights, and protocol metadata."""

    probabilities: np.ndarray
    pseudo_labels: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    keep_mask: np.ndarray
    sample_weights: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-locals

def target_confidence_weights(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: TargetConfidenceConfig | Mapping[str, Any] | None = None,
) -> TargetConfidenceResult:
    """Compute unlabeled target pseudo-label confidence and sample weights.

    Parameters
    ----------
    probabilities:
        Target probability rows from a source-trained model.  Rows are normalized
        internally.
    classes:
        Optional class labels in probability-column order.  If omitted, integer
        class indices are returned as pseudo-labels.
    config:
        Confidence thresholding and weighting settings.  A mapping is normalized
        through :func:`target_confidence_config`.

    Returns
    -------
    TargetConfidenceResult
        Normalized probabilities, pseudo-labels, confidence diagnostics, keep mask,
        sample weights, and Category-2 metadata.
    """

    cfg = target_confidence_config() if config is None else _coerce_config(config)
    probs = _probability_matrix(probabilities, epsilon=cfg.epsilon)
    class_values = _classes(classes, n_classes=probs.shape[1])
    pseudo_index = np.argmax(probs, axis=1)
    pseudo_labels = class_values[pseudo_index]
    confidence = np.max(probs, axis=1)
    margin = probability_margin(probs)
    entropy = normalized_entropy(probs, epsilon=cfg.epsilon)
    keep_mask = confidence >= cfg.confidence_threshold
    keep_mask = _ensure_min_keep(keep_mask, confidence, min_keep=cfg.min_keep)
    weights = _weights(cfg.weighting, confidence=confidence, margin=margin, entropy=entropy, keep_mask=keep_mask)
    if cfg.normalize_weights and np.any(weights > 0.0):
        weights = weights / float(np.mean(weights[weights > 0.0]))
    weights = np.where(keep_mask, weights, 0.0)
    metadata = {
        "target_confidence_weighting": True,
        "target_confidence_protocol": TARGET_CONFIDENCE_PROTOCOL,
        "target_confidence_protocol_category": TARGET_CONFIDENCE_CATEGORY,
        "target_confidence_uses_target_probabilities": True,
        "target_confidence_uses_target_features": False,
        "target_confidence_uses_target_labels": False,
        "target_confidence_valid_for_strict_source_only": False,
        "target_confidence_valid_for_unlabeled_target_adaptation": True,
        "target_confidence_valid_for_benchmark": False,
        "target_confidence_n_rows": int(probs.shape[0]),
        "target_confidence_n_classes": int(probs.shape[1]),
        "target_confidence_n_kept_rows": int(np.count_nonzero(keep_mask)),
        "target_confidence_threshold": float(cfg.confidence_threshold),
        "target_confidence_min_keep": int(cfg.min_keep),
        "target_confidence_weighting": cfg.weighting,
        "target_confidence_normalize_weights": bool(cfg.normalize_weights),
        "target_confidence_mean_confidence": float(np.mean(confidence)),
        "target_confidence_mean_entropy": float(np.mean(entropy)),
    }
    return TargetConfidenceResult(
        probabilities=probs.astype(np.float32, copy=False),
        pseudo_labels=pseudo_labels,
        confidence=confidence.astype(np.float32, copy=False),
        margin=margin.astype(np.float32, copy=False),
        entropy=entropy.astype(np.float32, copy=False),
        keep_mask=keep_mask.astype(bool, copy=False),
        sample_weights=weights.astype(np.float32, copy=False),
        metadata=metadata,
    )


def target_confidence_config(
    *,
    confidence_threshold: float | str = DEFAULT_CONFIDENCE_THRESHOLD,
    min_keep: int | str = DEFAULT_MIN_KEEP,
    weighting: str | None = "confidence",
    normalize_weights: bool | str | int | float = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> TargetConfidenceConfig:
    """Normalize target-confidence weighting options."""

    return TargetConfidenceConfig(
        confidence_threshold=_unit_interval_float(confidence_threshold, name="confidence_threshold"),
        min_keep=_nonnegative_int(min_keep, name="min_keep"),
        weighting=normalize_weighting_mode(weighting),
        normalize_weights=_bool_config(normalize_weights, name="normalize_weights"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_weighting_mode(value: str | None) -> str:
    """Normalize weighting-mode aliases."""

    normalized = "confidence" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "conf": "confidence",
        "probability": "confidence",
        "prob": "confidence",
        "entropy_inverse": "entropy",
        "low_entropy": "entropy",
        "hard": "mask",
        "binary": "mask",
    }.get(normalized, normalized)
    if normalized not in WEIGHTING_MODES:
        raise ValueError(f"Unknown weighting mode {value!r}. Available values: {', '.join(WEIGHTING_MODES)}.")
    return normalized


def probability_margin(probabilities: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return top-1 minus top-2 probability margin per row."""

    probs = _probability_matrix(probabilities, epsilon=DEFAULT_EPSILON)
    if probs.shape[1] == 1:
        return np.ones(probs.shape[0], dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def normalized_entropy(probabilities: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Return entropy normalized to [0, 1] for each probability row."""

    probs = _probability_matrix(probabilities, epsilon=epsilon)
    entropy = -np.sum(probs * np.log(np.maximum(probs, epsilon)), axis=1)
    return entropy / np.log(probs.shape[1])


def _weights(mode: str, *, confidence: np.ndarray, margin: np.ndarray, entropy: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    if mode == "confidence":
        return confidence.copy()
    if mode == "margin":
        return margin.copy()
    if mode == "entropy":
        return 1.0 - entropy
    if mode == "mask":
        return keep_mask.astype(float)
    raise ValueError(f"Unhandled weighting mode {mode!r}.")


def _ensure_min_keep(keep_mask: np.ndarray, confidence: np.ndarray, *, min_keep: int) -> np.ndarray:
    min_rows = min(int(min_keep), keep_mask.shape[0])
    if np.count_nonzero(keep_mask) >= min_rows:
        return keep_mask
    order = np.argsort(-confidence, kind="mergesort")[:min_rows]
    out = keep_mask.copy()
    out[order] = True
    return out


def _coerce_config(config: TargetConfidenceConfig | Mapping[str, Any]) -> TargetConfidenceConfig:
    if isinstance(config, TargetConfidenceConfig):
        return config
    return target_confidence_config(**dict(config))


def _classes(classes: Sequence[Any] | np.ndarray | None, *, n_classes: int) -> np.ndarray:
    if classes is None:
        return np.arange(n_classes, dtype=int)
    values = np.asarray(classes, dtype=object).reshape(-1)
    if values.shape[0] != n_classes:
        raise ValueError(f"classes must contain one value per probability column: {values.shape[0]} != {n_classes}.")
    if len(set(values.tolist())) != values.shape[0]:
        raise ValueError("classes must be unique.")
    return values


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    matrix = np.maximum(matrix, float(epsilon))
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _bool_config(value: bool | str | int | float, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean value.")


def _nonnegative_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _positive_or_zero_float(value, name=name)
    if parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _positive_or_zero_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
