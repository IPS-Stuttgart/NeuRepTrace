"""Source-only nearest-centroid decoder.

This module provides a small dependency-light baseline for strict cross-subject
decoding.  Class centroids are estimated from source rows and source labels only;
held-out rows are scored by scaled squared distance to the source centroids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_CENTROID_PROTOCOL = "strict_source_only_centroid_decoder"
SOURCE_CENTROID_CATEGORY = "1_strict_source_only"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceCentroidConfig:
    """Configuration for the source-centroid decoder."""

    temperature: float = DEFAULT_TEMPERATURE
    use_diagonal_scale: bool = True
    shrinkage: float = 0.0
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceCentroidResult:
    """Nearest-centroid predictions and provenance metadata."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    centroids: np.ndarray
    feature_scale: np.ndarray
    distances: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_centroid_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceCentroidConfig | Mapping[str, Any] | None = None,
) -> SourceCentroidResult:
    """Fit source class centroids and score test rows.

    This is a strict source-only baseline.  The API has no target adaptation
    inputs and no target-label inputs.
    """

    cfg = source_centroid_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    labels = _label_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    classes = np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=object)
    if classes.shape[0] < 2:
        raise ValueError("At least two source classes are required.")

    centroids, counts = _class_centroids(source, labels, classes=classes)
    if cfg.shrinkage > 0.0:
        global_mean = np.mean(source, axis=0, keepdims=True)
        centroids = (1.0 - cfg.shrinkage) * centroids + cfg.shrinkage * global_mean
    scale = _feature_scale(source, enabled=cfg.use_diagonal_scale, epsilon=cfg.epsilon)
    distances = _scaled_squared_distances(test, centroids, scale=scale)
    probabilities = _softmax(-distances / cfg.temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_test_rows=test.shape[0],
        feature_dim=source.shape[1],
        classes=classes,
        counts=counts,
    )
    return SourceCentroidResult(
        probabilities=probabilities.astype(np.float32, copy=False),
        predictions=predictions,
        classes=classes,
        centroids=centroids.astype(np.float32, copy=False),
        feature_scale=scale.astype(np.float32, copy=False),
        distances=distances.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_centroid_config(
    *,
    temperature: float | str = DEFAULT_TEMPERATURE,
    use_diagonal_scale: bool = True,
    shrinkage: float | str = 0.0,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceCentroidConfig:
    """Normalize public source-centroid options."""

    return SourceCentroidConfig(
        temperature=_positive_float(temperature, name="temperature"),
        use_diagonal_scale=bool(use_diagonal_scale),
        shrinkage=_unit_interval_float(shrinkage, name="shrinkage"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _coerce_config(config: SourceCentroidConfig | Mapping[str, Any]) -> SourceCentroidConfig:
    if isinstance(config, SourceCentroidConfig):
        return config
    return source_centroid_config(**dict(config))


def _class_centroids(features: np.ndarray, labels: np.ndarray, *, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.zeros((classes.shape[0], features.shape[1]), dtype=float)
    counts = np.zeros(classes.shape[0], dtype=int)
    for index, class_label in enumerate(classes.tolist()):
        mask = labels == class_label
        counts[index] = int(np.count_nonzero(mask))
        if counts[index] == 0:
            raise ValueError(f"No source rows available for class {class_label!r}.")
        centroids[index] = np.mean(features[mask], axis=0)
    return centroids, counts


def _feature_scale(features: np.ndarray, *, enabled: bool, epsilon: float) -> np.ndarray:
    if not enabled:
        return np.ones(features.shape[1], dtype=float)
    scale = np.std(features - np.mean(features, axis=0), axis=0, ddof=1 if features.shape[0] > 1 else 0)
    return np.maximum(scale, float(epsilon))


def _scaled_squared_distances(features: np.ndarray, centroids: np.ndarray, *, scale: np.ndarray) -> np.ndarray:
    normalized_features = features / scale
    normalized_centroids = centroids / scale
    diff = normalized_features[:, None, :] - normalized_centroids[None, :, :]
    return np.sum(diff * diff, axis=2)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _metadata(cfg: SourceCentroidConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, classes: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    return {
        "source_centroid_decoder": True,
        "source_centroid_protocol": SOURCE_CENTROID_PROTOCOL,
        "source_centroid_protocol_category": SOURCE_CENTROID_CATEGORY,
        "source_centroid_uses_source_features": True,
        "source_centroid_uses_source_labels": True,
        "source_centroid_uses_target_features_for_fitting": False,
        "source_centroid_uses_target_labels": False,
        "source_centroid_valid_for_strict_source_only": True,
        "source_centroid_valid_for_unlabeled_target_adaptation": True,
        "source_centroid_valid_for_benchmark": True,
        "source_centroid_n_source_rows": int(n_source_rows),
        "source_centroid_n_test_rows": int(n_test_rows),
        "source_centroid_feature_dim": int(feature_dim),
        "source_centroid_n_classes": int(classes.shape[0]),
        "source_centroid_temperature": float(cfg.temperature),
        "source_centroid_use_diagonal_scale": bool(cfg.use_diagonal_scale),
        "source_centroid_shrinkage": float(cfg.shrinkage),
        "source_centroid_class_counts": "|".join(f"{label}:{int(count)}" for label, count in zip(classes.tolist(), counts, strict=True)),
    }


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
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
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
