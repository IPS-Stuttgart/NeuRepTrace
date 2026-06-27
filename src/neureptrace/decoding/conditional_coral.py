"""Pseudo-label conditional CORAL alignment for Category-2 transfer.

The helpers in this module implement a class-conditional CORAL transform for
cross-subject M/EEG feature matrices.  Source class distributions are aligned
toward target pseudo-class distributions estimated from classifier predictions or
caller-supplied pseudo-labels/probabilities.

The public API intentionally has no target-label argument.  Target rows may be
used for pseudo-label adaptation, but held-out target labels must remain reserved
for scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONDITIONAL_CORAL_PROTOCOL = "unlabeled_target_pseudo_label_conditional_coral"
CONDITIONAL_CORAL_CATEGORY = "2_unlabeled_target_adaptive"
DEFAULT_CONDITIONAL_CORAL_REGULARIZATION = 1e-6
DEFAULT_CONDITIONAL_CORAL_MIN_TARGET_ROWS = 2


@dataclass(frozen=True, slots=True)
class ConditionalCoralConfig:
    """Configuration for pseudo-label conditional CORAL."""

    regularization: float = DEFAULT_CONDITIONAL_CORAL_REGULARIZATION
    min_target_rows_per_class: int = DEFAULT_CONDITIONAL_CORAL_MIN_TARGET_ROWS
    confidence_threshold: float = 0.0
    fallback: str = "global"
    center: bool = True
    random_state: int | None = 13


@dataclass(frozen=True, slots=True)
class CoralClassStats:
    """Class/domain feature statistics used by CORAL."""

    mean: np.ndarray
    covariance: np.ndarray
    n_rows: int


@dataclass(frozen=True, slots=True)
class ConditionalCoralResult:
    """Aligned train/test features and pseudo-label provenance."""

    train_features: np.ndarray
    test_features: np.ndarray
    classes: np.ndarray
    pseudo_labels: np.ndarray
    pseudo_confidence: np.ndarray
    source_stats: Mapping[Any, CoralClassStats]
    target_stats: Mapping[Any, CoralClassStats]
    global_source_stats: CoralClassStats
    global_target_stats: CoralClassStats
    used_fallback_classes: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_pseudo_label_conditional_coral(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    config: ConditionalCoralConfig | Mapping[str, Any] | None = None,
    estimator: BaseEstimator | None = None,
    target_pseudo_labels: Sequence[Any] | np.ndarray | None = None,
    target_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> ConditionalCoralResult:
    """Fit class-conditional CORAL using target pseudo-labels.

    Parameters
    ----------
    source_features, source_labels:
        Labeled source rows used to estimate source class distributions.
    target_features:
        Unlabeled target rows.  They are used to estimate pseudo-class target
        distributions, but not target labels.
    config:
        Conditional CORAL settings.  A mapping is normalized through
        :func:`conditional_coral_config`.
    estimator:
        Optional sklearn-style source classifier used when neither
        ``target_pseudo_labels`` nor ``target_probabilities`` are supplied.
    target_pseudo_labels:
        Optional classifier-generated target pseudo-labels.  These must be in the
        source class set and are not treated as true target labels.
    target_probabilities:
        Optional target class probabilities in source-class order.  Argmax labels
        become pseudo-labels and max probability becomes pseudo-confidence.

    Returns
    -------
    ConditionalCoralResult
        Source rows aligned class-wise toward pseudo-target class distributions;
        target rows are returned in their native feature space.

    Notes
    -----
    This is a Category-2 protocol.  The public API intentionally has no
    ``target_labels`` parameter.
    """

    cfg = conditional_coral_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")
    labels = np.asarray(source_labels, dtype=object).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError(f"source_labels must contain one value per source row: {labels.shape[0]} != {source.shape[0]}.")
    classes = np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=object)
    if classes.shape[0] < 2:
        raise ValueError("Conditional CORAL requires at least two source classes.")

    pseudo_labels, pseudo_confidence, pseudo_source = _resolve_pseudo_labels(
        source,
        labels,
        target,
        classes=classes,
        estimator=estimator,
        config=cfg,
        target_pseudo_labels=target_pseudo_labels,
        target_probabilities=target_probabilities,
    )
    confident_mask = pseudo_confidence >= cfg.confidence_threshold
    source_stats = {class_label: feature_stats(source[labels == class_label], regularization=cfg.regularization) for class_label in classes.tolist()}
    global_source = feature_stats(source, regularization=cfg.regularization)
    global_target = feature_stats(target[confident_mask] if np.any(confident_mask) else target, regularization=cfg.regularization)

    target_stats: dict[Any, CoralClassStats] = {}
    fallback_classes: list[Any] = []
    for class_label in classes.tolist():
        class_mask = (pseudo_labels == class_label) & confident_mask
        if np.count_nonzero(class_mask) >= cfg.min_target_rows_per_class:
            target_stats[class_label] = feature_stats(target[class_mask], regularization=cfg.regularization)
        elif cfg.fallback == "global":
            target_stats[class_label] = global_target
            fallback_classes.append(class_label)
        else:
            raise ValueError(
                "Target pseudo-class "
                f"{class_label!r} has {int(np.count_nonzero(class_mask))} rows, below min_target_rows_per_class={cfg.min_target_rows_per_class}."
            )

    aligned_source = np.empty_like(source, dtype=float)
    for class_label in classes.tolist():
        class_mask = labels == class_label
        aligned_source[class_mask] = coral_align_features(
            source[class_mask],
            source_stats=source_stats[class_label],
            target_stats=target_stats[class_label],
            center=cfg.center,
        )
    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_target_rows=target.shape[0],
        feature_dim=source.shape[1],
        n_classes=classes.shape[0],
        pseudo_source=pseudo_source,
        pseudo_labels=pseudo_labels,
        confident_mask=confident_mask,
        fallback_classes=tuple(fallback_classes),
    )
    return ConditionalCoralResult(
        train_features=aligned_source.astype(np.float32, copy=False),
        test_features=target.astype(np.float32, copy=False),
        classes=classes,
        pseudo_labels=pseudo_labels,
        pseudo_confidence=pseudo_confidence.astype(float, copy=False),
        source_stats=source_stats,
        target_stats=target_stats,
        global_source_stats=global_source,
        global_target_stats=global_target,
        used_fallback_classes=tuple(fallback_classes),
        metadata=metadata,
    )


def conditional_coral_config(
    *,
    regularization: float | str = DEFAULT_CONDITIONAL_CORAL_REGULARIZATION,
    min_target_rows_per_class: int | str = DEFAULT_CONDITIONAL_CORAL_MIN_TARGET_ROWS,
    confidence_threshold: float | str = 0.0,
    fallback: str = "global",
    center: bool = True,
    random_state: int | str | None = 13,
) -> ConditionalCoralConfig:
    """Normalize public conditional-CORAL options."""

    return ConditionalCoralConfig(
        regularization=_nonnegative_float(regularization, name="regularization"),
        min_target_rows_per_class=_positive_int(min_target_rows_per_class, name="min_target_rows_per_class"),
        confidence_threshold=_unit_interval_float(confidence_threshold, name="confidence_threshold"),
        fallback=normalize_conditional_coral_fallback(fallback),
        center=bool(center),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
    )


def normalize_conditional_coral_fallback(value: str | None) -> str:
    """Normalize fallback policy aliases."""

    normalized = "global" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"raise": "error", "strict": "error", "fail": "error"}.get(normalized, normalized)
    if normalized not in {"global", "error"}:
        raise ValueError("fallback must be 'global' or 'error'.")
    return normalized


def feature_stats(features: Sequence[Sequence[float]] | np.ndarray, *, regularization: float = DEFAULT_CONDITIONAL_CORAL_REGULARIZATION) -> CoralClassStats:
    """Return mean and regularized covariance for a feature matrix."""

    matrix = _feature_matrix(features, name="features")
    reg = _nonnegative_float(regularization, name="regularization")
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    if matrix.shape[0] <= 1:
        covariance = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    else:
        covariance = centered.T @ centered / float(matrix.shape[0] - 1)
    covariance = _nearest_spd(covariance + reg * np.eye(matrix.shape[1], dtype=float))
    return CoralClassStats(mean=mean.astype(float, copy=False), covariance=covariance, n_rows=int(matrix.shape[0]))


def coral_align_features(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_stats: CoralClassStats,
    target_stats: CoralClassStats,
    center: bool = True,
) -> np.ndarray:
    """Apply CORAL whitening/recoloring from source stats to target stats."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != source_stats.mean.shape[0] or matrix.shape[1] != target_stats.mean.shape[0]:
        raise ValueError("feature width must match source and target statistics.")
    source_inv_sqrt = _matrix_inv_sqrt_spd(source_stats.covariance)
    target_sqrt = _matrix_sqrt_spd(target_stats.covariance)
    centered = matrix - source_stats.mean
    recolored = centered @ source_inv_sqrt @ target_sqrt
    return recolored + target_stats.mean if center else recolored + source_stats.mean


def _resolve_pseudo_labels(
    source: np.ndarray,
    labels: np.ndarray,
    target: np.ndarray,
    *,
    classes: np.ndarray,
    estimator: BaseEstimator | None,
    config: ConditionalCoralConfig,
    target_pseudo_labels: Sequence[Any] | np.ndarray | None,
    target_probabilities: Sequence[Sequence[float]] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if target_pseudo_labels is not None and target_probabilities is not None:
        raise ValueError("Provide either target_pseudo_labels or target_probabilities, not both.")
    if target_probabilities is not None:
        probabilities = np.asarray(target_probabilities, dtype=float)
        if probabilities.shape != (target.shape[0], classes.shape[0]):
            raise ValueError(f"target_probabilities must have shape {(target.shape[0], classes.shape[0])}, got {probabilities.shape}.")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("target_probabilities must be finite and non-negative.")
        row_sums = probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("target_probabilities rows must have positive mass.")
        probabilities = probabilities / row_sums
        return classes[np.argmax(probabilities, axis=1)], np.max(probabilities, axis=1), "target_probabilities"
    if target_pseudo_labels is not None:
        pseudo = np.asarray(target_pseudo_labels, dtype=object).reshape(-1)
        if pseudo.shape[0] != target.shape[0]:
            raise ValueError(f"target_pseudo_labels must contain one value per target row: {pseudo.shape[0]} != {target.shape[0]}.")
        unknown = sorted({value for value in pseudo.tolist() if value not in set(classes.tolist())}, key=repr)
        if unknown:
            raise ValueError(f"target_pseudo_labels contain labels absent from source classes: {unknown}.")
        return pseudo, np.ones(target.shape[0], dtype=float), "target_pseudo_labels"
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)) if estimator is None else clone(estimator)
    model.fit(source, labels)
    pseudo = np.asarray(model.predict(target), dtype=object)
    confidence = np.ones(target.shape[0], dtype=float)
    if hasattr(model, "predict_proba"):
        raw_prob = np.asarray(model.predict_proba(target), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", classes), dtype=object)
        aligned = np.zeros((target.shape[0], classes.shape[0]), dtype=float)
        class_to_column = {class_label: index for index, class_label in enumerate(classes.tolist())}
        for column, class_label in enumerate(model_classes.tolist()):
            if class_label in class_to_column:
                aligned[:, class_to_column[class_label]] = raw_prob[:, column]
        row_sums = aligned.sum(axis=1, keepdims=True)
        aligned = np.divide(aligned, row_sums, out=np.full_like(aligned, 1.0 / classes.shape[0]), where=row_sums > 0.0)
        confidence = np.max(aligned, axis=1)
    return pseudo, confidence, "source_classifier"


def _metadata(
    cfg: ConditionalCoralConfig,
    *,
    n_source_rows: int,
    n_target_rows: int,
    feature_dim: int,
    n_classes: int,
    pseudo_source: str,
    pseudo_labels: np.ndarray,
    confident_mask: np.ndarray,
    fallback_classes: tuple[Any, ...],
) -> dict[str, Any]:
    unique, counts = np.unique(pseudo_labels.astype(str), return_counts=True)
    pseudo_counts = "|".join(f"{label}:{int(count)}" for label, count in zip(unique, counts, strict=True))
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
        "conditional_coral_pseudo_label_source": pseudo_source,
        "conditional_coral_pseudo_label_counts": pseudo_counts,
        "conditional_coral_confident_target_rows": int(np.count_nonzero(confident_mask)),
        "conditional_coral_regularization": float(cfg.regularization),
        "conditional_coral_min_target_rows_per_class": int(cfg.min_target_rows_per_class),
        "conditional_coral_confidence_threshold": float(cfg.confidence_threshold),
        "conditional_coral_fallback": cfg.fallback,
        "conditional_coral_fallback_classes": "|".join(str(value) for value in fallback_classes),
        "conditional_coral_center": bool(cfg.center),
    }


def _matrix_sqrt_spd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_nearest_spd(matrix))
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def _matrix_inv_sqrt_spd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_nearest_spd(matrix))
    values = np.maximum(values, np.finfo(float).eps)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _nearest_spd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    floor = np.finfo(float).eps * max(float(np.max(np.abs(values))) if values.size else 1.0, 1.0)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def _coerce_config(config: ConditionalCoralConfig | Mapping[str, Any]) -> ConditionalCoralConfig:
    if isinstance(config, ConditionalCoralConfig):
        return config
    return conditional_coral_config(**dict(config))


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
    integer = _normalize_integer(value, name=name)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_int(value: int | str, *, name: str) -> int:
    integer = _normalize_integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _normalize_integer(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
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
