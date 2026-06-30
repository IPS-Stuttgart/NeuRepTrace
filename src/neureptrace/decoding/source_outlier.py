"""Strict source-only outlier weighting helpers.

This module estimates per-source-row quality weights from distances to source
class centroids.  It is intended as a fold-local source-only preprocessing step:
rows that are far from their class centroid can be downweighted or filtered before
training an ordinary decoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_OUTLIER_PROTOCOL = "strict_source_only_class_distance_weighting"
SOURCE_OUTLIER_CATEGORY = "1_strict_source_only"
WEIGHT_MODES = ("binary", "linear", "soft")
THRESHOLD_MODES = ("quantile", "mad")
DEFAULT_QUANTILE = 0.95
DEFAULT_TEMPERATURE = 1.0
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceOutlierConfig:
    """Configuration for source-only class-distance weighting."""

    threshold_mode: str = "quantile"
    quantile: float = DEFAULT_QUANTILE
    mad_multiplier: float = 3.0
    weight_mode: str = "soft"
    temperature: float = DEFAULT_TEMPERATURE
    use_diagonal_scale: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceOutlierResult:
    """Per-row distances, weights, and inlier mask."""

    distances: np.ndarray
    sample_weights: np.ndarray
    inlier_mask: np.ndarray
    thresholds: Mapping[Any, float]
    classes: np.ndarray
    centroids: np.ndarray
    feature_scale: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def compute_source_outlier_weights(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    config: SourceOutlierConfig | Mapping[str, Any] | None = None,
) -> SourceOutlierResult:
    """Compute strict source-only outlier weights from class-centroid distances."""

    cfg = source_outlier_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    classes = np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=object)
    if classes.shape[0] < 2:
        raise ValueError("At least two source classes are required.")

    centroids = _class_centroids(features, labels, classes=classes)
    scale = _feature_scale(features, enabled=cfg.use_diagonal_scale, epsilon=cfg.epsilon)
    class_to_index = {class_label: index for index, class_label in enumerate(classes.tolist())}
    centroid_rows = np.vstack([centroids[class_to_index[label]] for label in labels.tolist()])
    distances = _scaled_l2(features, centroid_rows, scale=scale)
    thresholds = {
        class_label: _class_threshold(distances[labels == class_label], mode=cfg.threshold_mode, quantile=cfg.quantile, mad_multiplier=cfg.mad_multiplier)
        for class_label in classes.tolist()
    }
    threshold_rows = np.asarray([thresholds[label] for label in labels.tolist()], dtype=float)
    inlier_mask = distances <= threshold_rows
    weights = _weights(distances, threshold_rows, cfg=cfg)
    metadata = _metadata(cfg, labels=labels, classes=classes, distances=distances, inlier_mask=inlier_mask, thresholds=thresholds)
    return SourceOutlierResult(
        distances=distances.astype(np.float32, copy=False),
        sample_weights=weights.astype(np.float32, copy=False),
        inlier_mask=inlier_mask,
        thresholds=thresholds,
        classes=classes,
        centroids=np.vstack([centroids[label] for label in classes.tolist()]).astype(np.float32, copy=False),
        feature_scale=scale.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_outlier_config(
    *,
    threshold_mode: str | None = "quantile",
    quantile: float | str = DEFAULT_QUANTILE,
    mad_multiplier: float | str = 3.0,
    weight_mode: str | None = "soft",
    temperature: float | str = DEFAULT_TEMPERATURE,
    use_diagonal_scale: bool = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceOutlierConfig:
    """Normalize public source-outlier options."""

    return SourceOutlierConfig(
        threshold_mode=normalize_threshold_mode(threshold_mode),
        quantile=_unit_interval_float(quantile, name="quantile"),
        mad_multiplier=_positive_float(mad_multiplier, name="mad_multiplier"),
        weight_mode=normalize_weight_mode(weight_mode),
        temperature=_positive_float(temperature, name="temperature"),
        use_diagonal_scale=bool(use_diagonal_scale),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_threshold_mode(value: str | None) -> str:
    """Normalize threshold-mode aliases."""

    normalized = "quantile" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"percentile": "quantile", "median_absolute_deviation": "mad"}.get(normalized, normalized)
    if normalized not in THRESHOLD_MODES:
        raise ValueError(f"Unknown threshold_mode {value!r}.")
    return normalized


def normalize_weight_mode(value: str | None) -> str:
    """Normalize weight-mode aliases."""

    normalized = "soft" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"hard": "binary", "mask": "binary", "ramp": "linear", "exp": "soft", "exponential": "soft"}.get(normalized, normalized)
    if normalized not in WEIGHT_MODES:
        raise ValueError(f"Unknown weight_mode {value!r}.")
    return normalized


def _weights(distances: np.ndarray, thresholds: np.ndarray, *, cfg: SourceOutlierConfig) -> np.ndarray:
    safe_thresholds = np.maximum(thresholds, cfg.epsilon)
    ratio = distances / safe_thresholds
    if cfg.weight_mode == "binary":
        return (ratio <= 1.0).astype(float)
    if cfg.weight_mode == "linear":
        return np.clip(2.0 - ratio, 0.0, 1.0)
    if cfg.weight_mode == "soft":
        return np.exp(-np.maximum(ratio - 1.0, 0.0) / cfg.temperature)
    raise ValueError(f"Unhandled weight_mode {cfg.weight_mode!r}.")


def _class_threshold(distances: np.ndarray, *, mode: str, quantile: float, mad_multiplier: float) -> float:
    if distances.size < 1:
        raise ValueError("Each source class must contain at least one row.")
    if mode == "quantile":
        return float(np.quantile(distances, quantile))
    if mode == "mad":
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        return median + mad_multiplier * max(mad, np.finfo(float).eps)
    raise ValueError(f"Unhandled threshold_mode {mode!r}.")


def _class_centroids(features: np.ndarray, labels: np.ndarray, *, classes: np.ndarray) -> dict[Any, np.ndarray]:
    return {class_label: np.mean(features[labels == class_label], axis=0) for class_label in classes.tolist()}


def _feature_scale(features: np.ndarray, *, enabled: bool, epsilon: float) -> np.ndarray:
    if not enabled:
        return np.ones(features.shape[1], dtype=float)
    scale = np.std(features - np.mean(features, axis=0), axis=0, ddof=1 if features.shape[0] > 1 else 0)
    return np.maximum(scale, float(epsilon))


def _scaled_l2(left: np.ndarray, right: np.ndarray, *, scale: np.ndarray) -> np.ndarray:
    diff = (left - right) / scale
    return np.sqrt(np.sum(diff * diff, axis=1))


def _metadata(cfg: SourceOutlierConfig, *, labels: np.ndarray, classes: np.ndarray, distances: np.ndarray, inlier_mask: np.ndarray, thresholds: Mapping[Any, float]) -> dict[str, Any]:
    unique, counts = np.unique(labels.astype(str), return_counts=True)
    return {
        "source_outlier_weighting": True,
        "source_outlier_protocol": SOURCE_OUTLIER_PROTOCOL,
        "source_outlier_protocol_category": SOURCE_OUTLIER_CATEGORY,
        "source_outlier_uses_source_features": True,
        "source_outlier_uses_source_labels": True,
        "source_outlier_uses_heldout_features": False,
        "source_outlier_uses_heldout_labels": False,
        "source_outlier_valid_for_strict_source_only": True,
        "source_outlier_valid_for_benchmark": True,
        "source_outlier_n_rows": int(labels.shape[0]),
        "source_outlier_n_classes": int(classes.shape[0]),
        "source_outlier_threshold_mode": cfg.threshold_mode,
        "source_outlier_quantile": float(cfg.quantile),
        "source_outlier_mad_multiplier": float(cfg.mad_multiplier),
        "source_outlier_weight_mode": cfg.weight_mode,
        "source_outlier_temperature": float(cfg.temperature),
        "source_outlier_use_diagonal_scale": bool(cfg.use_diagonal_scale),
        "source_outlier_inlier_fraction": float(np.mean(inlier_mask)),
        "source_outlier_mean_distance": float(np.mean(distances)),
        "source_outlier_class_counts": "|".join(f"{label}:{int(count)}" for label, count in zip(unique, counts, strict=True)),
        "source_outlier_thresholds": "|".join(f"{label}:{float(thresholds[label]):.12g}" for label in classes.tolist()),
    }


def _coerce_config(config: SourceOutlierConfig | Mapping[str, Any]) -> SourceOutlierConfig:
    if isinstance(config, SourceOutlierConfig):
        return config
    return source_outlier_config(**dict(config))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    return vector


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
