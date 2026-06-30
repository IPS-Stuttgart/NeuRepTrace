"""Strict source-only k-nearest-neighbor decoder.

This module provides a dependency-light kNN baseline for cross-subject feature
decoding.  Neighbor search uses labeled source rows only.  Held-out rows are
scored by nearest-source labels and are never used for fitting or adaptation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_KNN_PROTOCOL = "strict_source_only_knn_decoder"
SOURCE_KNN_CATEGORY = "1_strict_source_only"
WEIGHT_MODES = ("uniform", "distance")
DEFAULT_K = 5
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceKNNConfig:
    """Configuration for source-only kNN decoding."""

    k: int | str = DEFAULT_K
    weights: str = "distance"
    standardize: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceKNNReference:
    """Source-only kNN reference data."""

    features: np.ndarray
    labels: np.ndarray
    classes: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    config: SourceKNNConfig


@dataclass(frozen=True, slots=True)
class SourceKNNResult:
    """kNN probabilities, predictions, and provenance metadata."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    reference: SourceKNNReference
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments
def fit_source_knn_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceKNNConfig | Mapping[str, Any] | None = None,
) -> SourceKNNResult:
    """Fit source-only kNN reference and score held-out rows."""

    cfg = source_knn_config() if config is None else _coerce_config(config)
    reference = fit_source_knn_reference(source_features=source_features, source_labels=source_labels, config=cfg)
    test = _feature_matrix(test_features, name="test_features")
    probabilities, neighbor_indices, neighbor_distances = predict_source_knn_probabilities(test, reference)
    predictions = reference.classes[np.argmax(probabilities, axis=1)]
    return SourceKNNResult(
        probabilities=probabilities.astype(np.float32, copy=False),
        predictions=predictions,
        classes=reference.classes,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances.astype(np.float32, copy=False),
        reference=reference,
        metadata=_metadata(reference, n_test_rows=test.shape[0]),
    )


def fit_source_knn_reference(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    config: SourceKNNConfig | Mapping[str, Any] | None = None,
) -> SourceKNNReference:
    """Store standardized source rows and labels for kNN scoring."""

    cfg = source_knn_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0])
    classes = _as_label_vector(_unique_labels_in_order(labels), name="classes")
    if classes.shape[0] < 2:
        raise ValueError("Source kNN requires at least two source classes.")
    if cfg.standardize:
        mean = np.mean(features, axis=0)
        scale = np.std(features - mean, axis=0, ddof=1 if features.shape[0] > 1 else 0)
        scale = np.maximum(scale, cfg.epsilon)
    else:
        mean = np.zeros(features.shape[1], dtype=float)
        scale = np.ones(features.shape[1], dtype=float)
    return SourceKNNReference(
        features=((features - mean) / scale).astype(np.float32, copy=False),
        labels=labels,
        classes=classes,
        mean=mean.astype(float, copy=False),
        scale=scale.astype(float, copy=False),
        config=cfg,
    )


def predict_source_knn_probabilities(
    test_features: Sequence[Sequence[float]] | np.ndarray,
    reference: SourceKNNReference,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict class probabilities from a fitted source-only kNN reference."""

    test = _feature_matrix(test_features, name="test_features")
    if test.shape[1] != reference.features.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{reference.features.shape[1]} != {test.shape[1]}."
        )
    prepared = (test - reference.mean) / reference.scale
    squared = _squared_euclidean(prepared, reference.features)
    distances = np.sqrt(np.maximum(squared, 0.0))
    k = min(_resolve_k(reference.config.k, n_source=reference.features.shape[0]), reference.features.shape[0])
    neighbor_indices = np.argsort(distances, axis=1, kind="mergesort")[:, :k]
    neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
    probabilities = np.zeros((test.shape[0], reference.classes.shape[0]), dtype=float)
    if reference.config.weights == "uniform":
        weights = np.ones_like(neighbor_distances, dtype=float)
    else:
        weights = 1.0 / np.maximum(neighbor_distances, reference.config.epsilon)
    for row in range(test.shape[0]):
        for local_col, source_index in enumerate(neighbor_indices[row]):
            class_col = _label_index(reference.classes, reference.labels[source_index])
            if class_col is None:
                raise ValueError(f"Source kNN reference contains unknown class label {reference.labels[source_index]!r}.")
            probabilities[row, class_col] += weights[row, local_col]
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    return probabilities, neighbor_indices.astype(int, copy=False), neighbor_distances.astype(float, copy=False)


def source_knn_config(
    *,
    k: int | str = DEFAULT_K,
    weights: str | None = "distance",
    standardize: bool | int | str = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceKNNConfig:
    """Normalize source-kNN options."""

    return SourceKNNConfig(
        k=k,
        weights=normalize_weight_mode(weights),
        standardize=_bool_value(standardize, name="standardize"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_weight_mode(value: str | None) -> str:
    """Normalize neighbor weighting aliases."""

    normalized = "distance" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"equal": "uniform", "flat": "uniform", "inverse_distance": "distance"}.get(normalized, normalized)
    if normalized not in WEIGHT_MODES:
        raise ValueError(f"Unknown source kNN weight mode {value!r}.")
    return normalized


def _coerce_config(config: SourceKNNConfig | Mapping[str, Any]) -> SourceKNNConfig:
    if isinstance(config, SourceKNNConfig):
        return config
    return source_knn_config(**dict(config))


def _metadata(reference: SourceKNNReference, *, n_test_rows: int) -> dict[str, Any]:
    return {
        "source_knn_decoder": True,
        "source_knn_protocol": SOURCE_KNN_PROTOCOL,
        "source_knn_protocol_category": SOURCE_KNN_CATEGORY,
        "source_knn_uses_source_features": True,
        "source_knn_uses_source_labels": True,
        "source_knn_uses_test_features_for_fitting": False,
        "source_knn_uses_test_labels": False,
        "source_knn_valid_for_strict_source_only": True,
        "source_knn_valid_for_benchmark": True,
        "source_knn_n_source_rows": int(reference.features.shape[0]),
        "source_knn_n_test_rows": int(n_test_rows),
        "source_knn_feature_dim": int(reference.features.shape[1]),
        "source_knn_n_classes": int(reference.classes.shape[0]),
        "source_knn_k": str(reference.config.k),
        "source_knn_weights": reference.config.weights,
        "source_knn_standardize": bool(reference.config.standardize),
        "source_knn_epsilon": float(reference.config.epsilon),
    }


def _resolve_k(value: int | str, *, n_source: int) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full"}:
            return int(n_source)
        parsed = float(text)
    else:
        parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError("k must be a positive integer, 'all', or 'full'.")
    return min(int(parsed), int(n_source))


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    labels = _as_label_vector(values, name="source_labels")
    if labels.shape[0] != expected_length:
        raise ValueError(f"source_labels must contain one value per source row: {labels.shape[0]} != {expected_length}.")
    return labels


def _as_label_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    """Return a one-dimensional object vector without expanding composite labels."""

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            items = array.tolist()
        elif array.ndim == 2 and array.shape[1] == 1:
            items = array.reshape(-1).tolist()
        elif array.ndim == 2:
            items = [tuple(row.tolist()) for row in array]
        else:
            raise ValueError(f"{name} must be one-dimensional or a two-dimensional composite-label matrix.")
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be one-dimensional.") from exc
    vector = np.empty(len(items), dtype=object)
    vector[:] = items
    return vector


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except Exception:
        return False
    if isinstance(equal, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(equal)


def _unique_labels_in_order(labels: np.ndarray) -> list[Any]:
    unique: list[Any] = []
    for label in labels.tolist():
        if not any(_labels_equal(label, seen) for seen in unique):
            unique.append(label)
    return unique


def _label_index(classes: np.ndarray, label: Any) -> int | None:
    for index, candidate in enumerate(classes.tolist()):
        if _labels_equal(label, candidate):
            return index
    return None


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
