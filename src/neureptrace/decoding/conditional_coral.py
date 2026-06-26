"""Pseudo-label class-conditional CORAL alignment for Category-2 transfer.

This module implements a dependency-light class-conditional CORAL transform for
cross-subject M/EEG feature matrices.  Source rows are grouped by source labels;
unlabeled target rows are grouped by classifier-generated pseudo-labels or target
probabilities; each source class is then covariance-aligned toward the matching
pseudo-target class.  Classes without enough pseudo-target support fall back to a
global unlabeled CORAL transform.

The public API intentionally has no target-label argument.  Target features and
pseudo-labels may be used for adaptation, but true target labels must remain
reserved for scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CONDITIONAL_CORAL_PROTOCOL = "pseudo_label_conditional_coral_alignment"
CONDITIONAL_CORAL_CATEGORY = "2_unlabeled_target_adaptive"
DEFAULT_MIN_TARGET_PER_CLASS = 2
DEFAULT_CONDITIONAL_CORAL_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class ConditionalCoralConfig:
    """Configuration for pseudo-label conditional CORAL."""

    min_target_per_class: int = DEFAULT_MIN_TARGET_PER_CLASS
    confidence_threshold: float = 0.0
    shrinkage: float = 0.0
    epsilon: float = DEFAULT_CONDITIONAL_CORAL_EPSILON
    fallback: str = "global"


@dataclass(frozen=True, slots=True)
class ConditionalCoralClassStats:
    """Per-class alignment statistics."""

    class_label: Any
    n_source: int
    n_target: int
    used_class_conditional: bool
    source_mean: np.ndarray
    target_mean: np.ndarray


@dataclass(frozen=True, slots=True)
class ConditionalCoralResult:
    """Aligned source features, target features, and protocol provenance."""

    train_features: np.ndarray
    test_features: np.ndarray
    classes: np.ndarray
    pseudo_labels: np.ndarray
    pseudo_confidence: np.ndarray
    class_stats: Mapping[Any, ConditionalCoralClassStats]
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_conditional_coral_alignment(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    target_pseudo_labels: Sequence[Any] | np.ndarray | None = None,
    target_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: ConditionalCoralConfig | Mapping[str, Any] | None = None,
) -> ConditionalCoralResult:
    """Fit pseudo-label conditional CORAL and transform source rows.

    Parameters
    ----------
    source_features, source_labels:
        Labeled source rows.  Source labels define the class-conditional source
        groups.
    target_features:
        Unlabeled target rows.  These rows are also returned unchanged as
        ``test_features`` because source rows are mapped into the target feature
        coordinate system.
    target_pseudo_labels:
        Optional classifier-generated target pseudo-labels.  These are not true
        target labels.
    target_probabilities:
        Optional target class-probability matrix.  If supplied,
        pseudo-labels/confidence are derived from the maximum probability.
    classes:
        Class order for ``target_probabilities``.  Defaults to source class order.
    config:
        Alignment settings.  Mappings are normalized with
        :func:`conditional_coral_config`.

    Returns
    -------
    ConditionalCoralResult
        Source rows aligned toward pseudo-target class distributions, target rows
        in native coordinates, and protocol metadata.

    Notes
    -----
    This is Category 2 / unlabeled target-adaptive.  The function intentionally
    has no ``target_labels`` parameter.
    """

    cfg = conditional_coral_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")
    labels = np.asarray(source_labels, dtype=object).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError(f"source_labels must contain one value per source row: {labels.shape[0]} != {source.shape[0]}.")
    source_classes = np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=object)
    if source_classes.shape[0] < 2:
        raise ValueError("conditional CORAL requires at least two source classes.")
    class_order = source_classes if classes is None else np.asarray(tuple(classes), dtype=object).reshape(-1)
    if class_order.shape[0] < 2:
        raise ValueError("classes must contain at least two class labels.")
    unknown_source = sorted({label for label in source_classes.tolist() if label not in set(class_order.tolist())}, key=repr)
    if unknown_source:
        raise ValueError(f"classes are missing source label(s): {unknown_source}.")

    pseudo, confidence, pseudo_source = _resolve_pseudo_labels(
        target_pseudo_labels=target_pseudo_labels,
        target_probabilities=target_probabilities,
        classes=class_order,
        n_target_rows=target.shape[0],
    )
    unknown_pseudo = sorted({label for label in pseudo.tolist() if label not in set(class_order.tolist())}, key=repr)
    if unknown_pseudo:
        raise ValueError(f"target_pseudo_labels contain labels absent from classes: {unknown_pseudo}.")

    global_source_stats = _domain_stats(source, shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
    global_target_stats = _domain_stats(target, shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
    global_aligned = _coral_transform(source, global_source_stats, global_target_stats, epsilon=cfg.epsilon)
    aligned = np.empty_like(source, dtype=float)
    class_stats: dict[Any, ConditionalCoralClassStats] = {}
    selected_target = confidence >= cfg.confidence_threshold
    n_conditional = 0

    for class_label in source_classes.tolist():
        source_mask = labels == class_label
        target_mask = (pseudo == class_label) & selected_target
        source_class = source[source_mask]
        n_source = int(np.sum(source_mask))
        n_target = int(np.sum(target_mask))
        use_conditional = n_target >= cfg.min_target_per_class and n_source >= 1
        if use_conditional:
            source_stats = _domain_stats(source_class, shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
            target_stats = _domain_stats(target[target_mask], shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
            aligned[source_mask] = _coral_transform(source_class, source_stats, target_stats, epsilon=cfg.epsilon)
            n_conditional += 1
            used_source_stats = source_stats
            used_target_stats = target_stats
        elif cfg.fallback == "identity":
            aligned[source_mask] = source_class
            used_source_stats = _domain_stats(source_class, shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
            used_target_stats = used_source_stats
        else:
            aligned[source_mask] = global_aligned[source_mask]
            used_source_stats = _domain_stats(source_class, shrinkage=cfg.shrinkage, epsilon=cfg.epsilon)
            used_target_stats = global_target_stats
        class_stats[class_label] = ConditionalCoralClassStats(
            class_label=class_label,
            n_source=n_source,
            n_target=n_target,
            used_class_conditional=bool(use_conditional),
            source_mean=used_source_stats.mean.astype(float, copy=False),
            target_mean=used_target_stats.mean.astype(float, copy=False),
        )

    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_target_rows=target.shape[0],
        feature_dim=source.shape[1],
        n_classes=source_classes.shape[0],
        n_conditional_classes=n_conditional,
        pseudo_source=pseudo_source,
        confidence=confidence,
        pseudo=pseudo,
        class_stats=class_stats,
    )
    return ConditionalCoralResult(
        train_features=aligned.astype(np.float32, copy=False),
        test_features=target.astype(np.float32, copy=False),
        classes=source_classes,
        pseudo_labels=pseudo,
        pseudo_confidence=confidence.astype(float, copy=False),
        class_stats=class_stats,
        metadata=metadata,
    )


def conditional_coral_config(
    *,
    min_target_per_class: int | str = DEFAULT_MIN_TARGET_PER_CLASS,
    confidence_threshold: float | str = 0.0,
    shrinkage: float | str = 0.0,
    epsilon: float | str = DEFAULT_CONDITIONAL_CORAL_EPSILON,
    fallback: str | None = "global",
) -> ConditionalCoralConfig:
    """Normalize public conditional-CORAL options."""

    return ConditionalCoralConfig(
        min_target_per_class=_positive_int(min_target_per_class, name="min_target_per_class"),
        confidence_threshold=_unit_interval_float(confidence_threshold, name="confidence_threshold"),
        shrinkage=_unit_interval_float(shrinkage, name="shrinkage"),
        epsilon=_positive_float(epsilon, name="epsilon"),
        fallback=normalize_conditional_coral_fallback(fallback),
    )


def normalize_conditional_coral_fallback(value: str | None) -> str:
    """Normalize fallback behavior for classes without enough pseudo-target rows."""

    normalized = "global" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"none": "identity", "off": "identity", "no_alignment": "identity", "global_coral": "global"}.get(normalized, normalized)
    if normalized not in {"global", "identity"}:
        raise ValueError("conditional CORAL fallback must be 'global' or 'identity'.")
    return normalized


def _resolve_pseudo_labels(
    *,
    target_pseudo_labels: Sequence[Any] | np.ndarray | None,
    target_probabilities: Sequence[Sequence[float]] | np.ndarray | None,
    classes: np.ndarray,
    n_target_rows: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if target_probabilities is not None:
        probabilities = _probability_matrix(target_probabilities, n_rows=n_target_rows, n_classes=classes.shape[0])
        indices = np.argmax(probabilities, axis=1)
        return classes[indices].astype(object, copy=False), np.max(probabilities, axis=1), "target_probabilities"
    if target_pseudo_labels is None:
        raise ValueError("Provide target_probabilities or classifier-generated target_pseudo_labels.")
    pseudo = np.asarray(target_pseudo_labels, dtype=object).reshape(-1)
    if pseudo.shape[0] != n_target_rows:
        raise ValueError(f"target_pseudo_labels must contain one value per target row: {pseudo.shape[0]} != {n_target_rows}.")
    return pseudo, np.ones(n_target_rows, dtype=float), "target_pseudo_labels"


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, n_rows: int, n_classes: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (n_rows, n_classes):
        raise ValueError(f"target_probabilities must have shape {(n_rows, n_classes)}, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("target_probabilities must contain finite non-negative values.")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("target_probabilities rows must have positive mass.")
    return matrix / row_sums


def _coerce_config(config: ConditionalCoralConfig | Mapping[str, Any]) -> ConditionalCoralConfig:
    if isinstance(config, ConditionalCoralConfig):
        return config
    return conditional_coral_config(**dict(config))


@dataclass(frozen=True, slots=True)
class _DomainStats:
    mean: np.ndarray
    covariance: np.ndarray
    sqrt: np.ndarray
    inv_sqrt: np.ndarray


def _domain_stats(features: np.ndarray, *, shrinkage: float, epsilon: float) -> _DomainStats:
    matrix = _feature_matrix(features, name="domain_features")
    mean = np.mean(matrix, axis=0)
    covariance = _regularized_covariance(matrix, shrinkage=shrinkage, epsilon=epsilon)
    sqrt = _matrix_power(covariance, 0.5, epsilon=epsilon)
    inv_sqrt = _matrix_power(covariance, -0.5, epsilon=epsilon)
    return _DomainStats(mean=mean, covariance=covariance, sqrt=sqrt, inv_sqrt=inv_sqrt)


def _coral_transform(features: np.ndarray, source_stats: _DomainStats, target_stats: _DomainStats, *, epsilon: float) -> np.ndarray:
    del epsilon
    centered = features - source_stats.mean
    return centered @ source_stats.inv_sqrt @ target_stats.sqrt + target_stats.mean


def _regularized_covariance(features: np.ndarray, *, shrinkage: float, epsilon: float) -> np.ndarray:
    n_rows, n_features = features.shape
    if n_rows <= 1:
        covariance = np.eye(n_features, dtype=float)
    else:
        centered = features - np.mean(features, axis=0, keepdims=True)
        covariance = centered.T @ centered / float(n_rows - 1)
    trace_scale = float(np.trace(covariance) / max(1, n_features))
    identity = np.eye(n_features, dtype=float) * max(trace_scale, float(epsilon))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * identity
    covariance = 0.5 * (covariance + covariance.T)
    covariance += float(epsilon) * np.eye(n_features, dtype=float)
    return covariance


def _matrix_power(matrix: np.ndarray, power: float, *, epsilon: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values = np.maximum(values, float(epsilon))
    return (vectors * (values**power)) @ vectors.T


def _metadata(
    cfg: ConditionalCoralConfig,
    *,
    n_source_rows: int,
    n_target_rows: int,
    feature_dim: int,
    n_classes: int,
    n_conditional_classes: int,
    pseudo_source: str,
    confidence: np.ndarray,
    pseudo: np.ndarray,
    class_stats: Mapping[Any, ConditionalCoralClassStats],
) -> dict[str, Any]:
    unique, counts = np.unique(pseudo.astype(str), return_counts=True)
    pseudo_counts = "|".join(f"{label}:{int(count)}" for label, count in zip(unique, counts, strict=True))
    conditional_counts = "|".join(f"{label}:{stats.n_target}:{int(stats.used_class_conditional)}" for label, stats in class_stats.items())
    return {
        "conditional_coral": True,
        "conditional_coral_protocol": CONDITIONAL_CORAL_PROTOCOL,
        "conditional_coral_protocol_category": CONDITIONAL_CORAL_CATEGORY,
        "conditional_coral_uses_source_features": True,
        "conditional_coral_uses_source_labels": True,
        "conditional_coral_uses_target_features": True,
        "conditional_coral_uses_target_labels": False,
        "conditional_coral_uses_target_pseudo_labels": True,
        "conditional_coral_valid_for_strict_source_only": False,
        "conditional_coral_valid_for_unlabeled_target_adaptation": True,
        "conditional_coral_valid_for_benchmark": False,
        "conditional_coral_n_source_rows": int(n_source_rows),
        "conditional_coral_n_target_rows": int(n_target_rows),
        "conditional_coral_feature_dim": int(feature_dim),
        "conditional_coral_n_classes": int(n_classes),
        "conditional_coral_n_conditional_classes": int(n_conditional_classes),
        "conditional_coral_pseudo_label_source": pseudo_source,
        "conditional_coral_pseudo_label_counts": pseudo_counts,
        "conditional_coral_class_target_counts": conditional_counts,
        "conditional_coral_min_target_per_class": int(cfg.min_target_per_class),
        "conditional_coral_confidence_threshold": float(cfg.confidence_threshold),
        "conditional_coral_mean_pseudo_confidence": float(np.mean(confidence)) if confidence.size else np.nan,
        "conditional_coral_shrinkage": float(cfg.shrinkage),
        "conditional_coral_epsilon": float(cfg.epsilon),
        "conditional_coral_fallback": cfg.fallback,
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _float_value(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed
