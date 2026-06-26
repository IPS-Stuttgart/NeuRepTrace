"""Class-conditional CORAL using pseudo classes for unlabeled target adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CONDITIONAL_CORAL_PROTOCOL = "pseudo_conditional_coral_alignment"
CONDITIONAL_CORAL_CATEGORY = "2_unlabeled_target_adaptive"


@dataclass(frozen=True, slots=True)
class ConditionalCoralConfig:
    confidence_threshold: float = 0.0
    min_target_per_class: int = 2
    shrinkage: float = 0.05
    epsilon: float = 1e-6
    fallback_to_global: bool = True


@dataclass(frozen=True, slots=True)
class CoralStats:
    mean: np.ndarray
    covariance: np.ndarray
    n_rows: int


@dataclass(frozen=True, slots=True)
class ConditionalCoralResult:
    train_features: np.ndarray
    test_features: np.ndarray
    classes: np.ndarray
    pseudo_labels: np.ndarray
    pseudo_confidence: np.ndarray
    class_counts: Mapping[Any, int]
    metadata: dict[str, Any] = field(default_factory=dict)


def conditional_coral_config(
    *,
    confidence_threshold: float | str = 0.0,
    min_target_per_class: int | str = 2,
    shrinkage: float | str = 0.05,
    epsilon: float | str = 1e-6,
    fallback_to_global: bool = True,
) -> ConditionalCoralConfig:
    return ConditionalCoralConfig(
        confidence_threshold=_unit_interval(confidence_threshold, name="confidence_threshold"),
        min_target_per_class=_positive_int(min_target_per_class, name="min_target_per_class"),
        shrinkage=_unit_interval(shrinkage, name="shrinkage"),
        epsilon=_positive_float(epsilon, name="epsilon"),
        fallback_to_global=bool(fallback_to_global),
    )


def fit_pseudo_conditional_coral(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    pseudo_labels: Sequence[Any] | np.ndarray | None = None,
    pseudo_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: ConditionalCoralConfig | Mapping[str, Any] | None = None,
) -> ConditionalCoralResult:
    """Transform source rows with class-wise CORAL maps estimated from pseudo classes."""

    cfg = conditional_coral_config() if config is None else _coerce_config(config)
    source = _matrix(source_features, name="source_features")
    target = _matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have the same feature width.")
    labels = np.asarray(source_labels, dtype=object).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError("source_labels must contain one value per source row.")
    class_values = _classes(labels, classes)
    pseudo, confidence = _pseudo(target, pseudo_labels=pseudo_labels, pseudo_probabilities=pseudo_probabilities, classes=class_values)

    global_target = _stats(target, shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
    aligned = np.empty_like(source, dtype=float)
    counts: dict[Any, int] = {}
    used = 0
    for class_value in class_values.tolist():
        source_mask = labels == class_value
        class_source = _stats(source[source_mask], shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
        target_mask = (pseudo == class_value) & (confidence >= cfg.confidence_threshold)
        counts[class_value] = int(np.count_nonzero(target_mask))
        if counts[class_value] >= cfg.min_target_per_class:
            class_target = _stats(target[target_mask], shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
            aligned[source_mask] = coral_transform(source[source_mask], class_source, class_target, epsilon=cfg.epsilon)
            used += 1
        elif cfg.fallback_to_global:
            aligned[source_mask] = coral_transform(source[source_mask], class_source, global_target, epsilon=cfg.epsilon)
        else:
            aligned[source_mask] = source[source_mask]

    return ConditionalCoralResult(
        train_features=aligned.astype(np.float32, copy=False),
        test_features=target.astype(np.float32, copy=False),
        classes=class_values,
        pseudo_labels=pseudo,
        pseudo_confidence=confidence.astype(np.float32, copy=False),
        class_counts=counts,
        metadata=_metadata(cfg, source.shape, target.shape, class_values.shape[0], used, counts),
    )


def coral_transform(features: Sequence[Sequence[float]] | np.ndarray, source_stats: CoralStats, target_stats: CoralStats, *, epsilon: float = 1e-6) -> np.ndarray:
    matrix = _matrix(features, name="features")
    transform = _inv_sqrt(source_stats.covariance, epsilon=epsilon) @ _sqrt(target_stats.covariance, epsilon=epsilon)
    return (matrix - source_stats.mean) @ transform + target_stats.mean


def _coerce_config(config: ConditionalCoralConfig | Mapping[str, Any]) -> ConditionalCoralConfig:
    if isinstance(config, ConditionalCoralConfig):
        return config
    return conditional_coral_config(**dict(config))


def _classes(labels: np.ndarray, classes: Sequence[Any] | np.ndarray | None) -> np.ndarray:
    values = np.asarray(tuple(dict.fromkeys(labels.tolist())) if classes is None else tuple(classes), dtype=object)
    if values.shape[0] < 2:
        raise ValueError("At least two classes are required.")
    missing = {label for label in labels.tolist() if label not in set(values.tolist())}
    if missing:
        raise ValueError(f"classes is missing source label(s): {sorted(missing, key=repr)}")
    return values


def _pseudo(target: np.ndarray, *, pseudo_labels: Sequence[Any] | np.ndarray | None, pseudo_probabilities: Sequence[Sequence[float]] | np.ndarray | None, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pseudo_labels is not None and pseudo_probabilities is not None:
        raise ValueError("Supply pseudo_labels or pseudo_probabilities, not both.")
    if pseudo_probabilities is not None:
        probabilities = _probabilities(pseudo_probabilities, rows=target.shape[0], cols=classes.shape[0])
        return classes[np.argmax(probabilities, axis=1)].astype(object, copy=False), np.max(probabilities, axis=1)
    if pseudo_labels is None:
        raise ValueError("pseudo_labels or pseudo_probabilities is required.")
    pseudo = np.asarray(pseudo_labels, dtype=object).reshape(-1)
    if pseudo.shape[0] != target.shape[0]:
        raise ValueError("pseudo_labels must contain one value per target row.")
    unknown = {label for label in pseudo.tolist() if label not in set(classes.tolist())}
    if unknown:
        raise ValueError(f"pseudo_labels contain unknown class values: {sorted(unknown, key=repr)}")
    return pseudo, np.ones(target.shape[0], dtype=float)


def _stats(features: np.ndarray, *, shrinkage: float, epsilon: float) -> CoralStats:
    matrix = _matrix(features, name="features")
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    if matrix.shape[0] <= 1:
        covariance = np.eye(matrix.shape[1], dtype=float)
    else:
        covariance = centered.T @ centered / float(matrix.shape[0] - 1)
    trace = float(np.trace(covariance) / max(1, covariance.shape[0]))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.eye(covariance.shape[0]) * max(trace, epsilon)
    return CoralStats(mean=mean, covariance=_nearest_spd(covariance, epsilon=epsilon), n_rows=int(matrix.shape[0]))


def _sqrt(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(_nearest_spd(matrix, epsilon=epsilon))
    return vectors @ np.diag(np.sqrt(np.maximum(values, epsilon))) @ vectors.T


def _inv_sqrt(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(_nearest_spd(matrix, epsilon=epsilon))
    return vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, epsilon))) @ vectors.T


def _nearest_spd(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    return vectors @ np.diag(np.maximum(values, float(epsilon))) @ vectors.T


def _metadata(cfg: ConditionalCoralConfig, source_shape: tuple[int, int], target_shape: tuple[int, int], n_classes: int, n_used_classes: int, counts: Mapping[Any, int]) -> dict[str, Any]:
    return {
        "conditional_coral": True,
        "conditional_coral_protocol": CONDITIONAL_CORAL_PROTOCOL,
        "conditional_coral_protocol_category": CONDITIONAL_CORAL_CATEGORY,
        "conditional_coral_uses_source_features": True,
        "conditional_coral_uses_source_labels": True,
        "conditional_coral_uses_target_features": True,
        "conditional_coral_uses_target_y": False,
        "conditional_coral_uses_pseudo_classes": True,
        "conditional_coral_valid_for_strict_source_only": False,
        "conditional_coral_valid_for_unlabeled_target_adaptation": True,
        "conditional_coral_valid_for_benchmark": False,
        "conditional_coral_n_source_rows": int(source_shape[0]),
        "conditional_coral_n_target_rows": int(target_shape[0]),
        "conditional_coral_feature_dim": int(source_shape[1]),
        "conditional_coral_n_classes": int(n_classes),
        "conditional_coral_n_target_supported_classes": int(n_used_classes),
        "conditional_coral_confidence_threshold": float(cfg.confidence_threshold),
        "conditional_coral_min_target_per_class": int(cfg.min_target_per_class),
        "conditional_coral_shrinkage": float(cfg.shrinkage),
        "conditional_coral_epsilon": float(cfg.epsilon),
        "conditional_coral_fallback_to_global": bool(cfg.fallback_to_global),
        "conditional_coral_pseudo_counts": "|".join(f"{label}:{int(count)}" for label, count in counts.items()),
    }


def _matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _probabilities(values: Sequence[Sequence[float]] | np.ndarray, *, rows: int, cols: int) -> np.ndarray:
    matrix = _matrix(values, name="pseudo_probabilities")
    if matrix.shape != (rows, cols):
        raise ValueError(f"pseudo_probabilities must have shape {(rows, cols)}, got {matrix.shape}.")
    if np.any(matrix < 0.0):
        raise ValueError("pseudo_probabilities must be non-negative.")
    total = np.sum(matrix, axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("pseudo_probabilities rows must have positive mass.")
    return matrix / total


def _positive_int(value: int | str, *, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _unit_interval(value: float | str, *, name: str) -> float:
    parsed = _float(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed
