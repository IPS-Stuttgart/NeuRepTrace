"""Strict source-only class-prior adjustment helpers.

This module estimates class priors from source labels only and can use those
priors to reweight probability rows.  It is a Protocol-1 post-processing helper:
held-out probability rows may be transformed, but held-out labels and held-out
features are not used for fitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_PRIOR_PROTOCOL = "strict_source_only_class_prior_adjustment"
SOURCE_PRIOR_CATEGORY = "1_strict_source_only"
TARGET_PRIORS = ("uniform", "source")
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourcePriorConfig:
    """Configuration for source-only prior adjustment."""

    target_prior: str = "uniform"
    smoothing: float = 0.0
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourcePriorAdjustmentResult:
    """Adjusted probabilities and source-only provenance."""

    probabilities: np.ndarray
    source_prior: np.ndarray
    target_prior: np.ndarray
    classes: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_source_class_prior(
    source_labels: Sequence[Any] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    smoothing: float | str = 0.0,
    epsilon: float | str = DEFAULT_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a class prior distribution from source labels only."""

    labels = np.asarray(source_labels, dtype=object).reshape(-1)
    if labels.shape[0] < 1:
        raise ValueError("source_labels must contain at least one value.")
    class_values = _classes(labels, classes)
    smooth = _nonnegative_float(smoothing, name="smoothing")
    counts = np.asarray([np.count_nonzero(labels == label) for label in class_values.tolist()], dtype=float)
    counts = counts + smooth
    prior = _normalize_probability_vector(counts, epsilon=_positive_float(epsilon, name="epsilon"))
    return prior.astype(np.float32, copy=False), class_values


def adjust_probabilities_to_source_prior(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_labels: Sequence[Any] | np.ndarray,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: SourcePriorConfig | Mapping[str, Any] | None = None,
) -> SourcePriorAdjustmentResult:
    """Reweight probability rows using priors estimated from source labels.

    ``probabilities`` are assumed to be in ``classes`` order.  If ``classes`` is
    omitted, the order is the first-occurrence order in ``source_labels``.
    """

    cfg = source_prior_config() if config is None else _coerce_config(config)
    source_prior, class_values = estimate_source_class_prior(source_labels, classes=classes, smoothing=cfg.smoothing, epsilon=cfg.epsilon)
    prob = _probability_matrix(probabilities, n_classes=class_values.shape[0], epsilon=cfg.epsilon)
    target_prior = _target_prior(source_prior, target_prior=cfg.target_prior, epsilon=cfg.epsilon)
    weights = target_prior / np.maximum(source_prior, cfg.epsilon)
    adjusted = _normalize_probability_rows(prob * weights[None, :], epsilon=cfg.epsilon)
    metadata = _metadata(cfg, n_rows=prob.shape[0], n_classes=class_values.shape[0])
    return SourcePriorAdjustmentResult(
        probabilities=adjusted.astype(np.float32, copy=False),
        source_prior=source_prior.astype(np.float32, copy=False),
        target_prior=target_prior.astype(np.float32, copy=False),
        classes=class_values,
        metadata=metadata,
    )


def source_prior_config(
    *,
    target_prior: str | None = "uniform",
    smoothing: float | str = 0.0,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourcePriorConfig:
    """Normalize source-prior adjustment options."""

    return SourcePriorConfig(
        target_prior=normalize_target_prior(target_prior),
        smoothing=_nonnegative_float(smoothing, name="smoothing"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_target_prior(value: str | None) -> str:
    """Normalize target-prior aliases."""

    normalized = "uniform" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"balanced": "uniform", "flat": "uniform", "empirical": "source", "source_prior": "source"}.get(normalized, normalized)
    if normalized not in TARGET_PRIORS:
        raise ValueError(f"Unknown target_prior {value!r}. Available priors: {', '.join(TARGET_PRIORS)}.")
    return normalized


def _coerce_config(config: SourcePriorConfig | Mapping[str, Any]) -> SourcePriorConfig:
    if isinstance(config, SourcePriorConfig):
        return config
    return source_prior_config(**dict(config))


def _target_prior(source_prior: np.ndarray, *, target_prior: str, epsilon: float) -> np.ndarray:
    if target_prior == "source":
        return source_prior.copy()
    if target_prior == "uniform":
        return np.full(source_prior.shape[0], 1.0 / source_prior.shape[0], dtype=float)
    raise ValueError(f"Unhandled target_prior {target_prior!r}.")


def _classes(labels: np.ndarray, classes: Sequence[Any] | np.ndarray | None) -> np.ndarray:
    if classes is None:
        return np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=object)
    class_values = np.asarray(classes, dtype=object).reshape(-1)
    if class_values.shape[0] < 1:
        raise ValueError("classes must contain at least one value.")
    if len(set(class_values.tolist())) != class_values.shape[0]:
        raise ValueError("classes must be unique.")
    unknown = sorted({label for label in labels.tolist() if label not in set(class_values.tolist())}, key=repr)
    if unknown:
        raise ValueError(f"source_labels contain labels absent from classes: {unknown}.")
    return class_values


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, n_classes: int, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != n_classes:
        raise ValueError(f"probabilities must have shape n_rows x {n_classes}.")
    return _normalize_probability_rows(matrix, epsilon=epsilon)


def _normalize_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probability rows must be finite and non-negative.")
    matrix = np.maximum(matrix, float(epsilon))
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _normalize_probability_vector(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape[0] < 1 or not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("prior values must be finite and non-negative.")
    vector = np.maximum(vector, float(epsilon))
    return vector / float(np.sum(vector))


def _metadata(cfg: SourcePriorConfig, *, n_rows: int, n_classes: int) -> dict[str, Any]:
    return {
        "source_prior_adjustment": True,
        "source_prior_protocol": SOURCE_PRIOR_PROTOCOL,
        "source_prior_protocol_category": SOURCE_PRIOR_CATEGORY,
        "source_prior_target_prior": cfg.target_prior,
        "source_prior_uses_source_labels": True,
        "source_prior_uses_heldout_features": False,
        "source_prior_uses_heldout_labels": False,
        "source_prior_valid_for_strict_source_only": True,
        "source_prior_valid_for_benchmark": True,
        "source_prior_n_probability_rows": int(n_rows),
        "source_prior_n_classes": int(n_classes),
        "source_prior_smoothing": float(cfg.smoothing),
        "source_prior_epsilon": float(cfg.epsilon),
    }


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed
