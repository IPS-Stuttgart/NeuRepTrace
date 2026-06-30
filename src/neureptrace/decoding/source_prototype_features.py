"""Strict source-only prototype-distance feature helpers.

This module fits class prototypes from source rows and source labels only, then
represents source and held-out rows by their distances to those frozen source
prototypes.  It is a Protocol-1 preprocessing helper: held-out rows are
transformed but never used to fit prototypes, scales, or class order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask

SOURCE_PROTOTYPE_FEATURES_PROTOCOL = "strict_source_only_prototype_distance_features"
SOURCE_PROTOTYPE_FEATURES_CATEGORY = "1_strict_source_only"
PROTOTYPE_METRICS = ("squared_euclidean", "euclidean", "cosine")
PROTOTYPE_OUTPUTS = ("distance", "negative_distance", "rbf_similarity")
DEFAULT_TEMPERATURE = 1.0
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourcePrototypeFeatureConfig:
    """Configuration for source-only prototype-distance features."""

    metric: str = "squared_euclidean"
    output: str = "distance"
    use_diagonal_scale: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourcePrototypeFeatureResult:
    """Prototype-distance features and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    classes: np.ndarray
    prototypes: np.ndarray
    feature_scale: np.ndarray
    source_counts: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals
def fit_source_prototype_features(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourcePrototypeFeatureConfig | Mapping[str, Any] | None = None,
) -> SourcePrototypeFeatureResult:
    """Fit source class prototypes and transform source/test rows.

    Parameters
    ----------
    source_features, source_labels:
        Source rows and labels used to estimate class prototypes.
    test_features:
        Held-out rows transformed with the fixed source prototypes.
    config:
        Prototype feature options.  A mapping is normalized through
        :func:`source_prototype_feature_config`.
    """

    cfg = source_prototype_feature_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    labels = _label_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    classes, counts = label_counts(labels)
    if classes.shape[0] < 2:
        raise ValueError("Prototype features require at least two source classes.")
    prototypes = class_prototypes(source, labels, classes=classes)
    scale = _feature_scale(source, enabled=cfg.use_diagonal_scale, epsilon=cfg.epsilon)
    train = prototype_distance_features(source, prototypes, metric=cfg.metric, output=cfg.output, feature_scale=scale, temperature=cfg.temperature, epsilon=cfg.epsilon)
    test_out = prototype_distance_features(test, prototypes, metric=cfg.metric, output=cfg.output, feature_scale=scale, temperature=cfg.temperature, epsilon=cfg.epsilon)
    return SourcePrototypeFeatureResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        classes=classes,
        prototypes=prototypes.astype(np.float32, copy=False),
        feature_scale=scale.astype(np.float32, copy=False),
        source_counts=counts.astype(int, copy=False),
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], n_classes=classes.shape[0], counts=counts),
    )


def class_prototypes(source_features: Sequence[Sequence[float]] | np.ndarray, source_labels: Sequence[Any] | np.ndarray, *, classes: Sequence[Any] | np.ndarray | None = None) -> np.ndarray:
    """Return class means from source rows only."""

    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    class_values = label_counts(labels)[0] if classes is None else _object_vector(np.asarray(classes, dtype=object).reshape(-1).tolist())
    if class_values.shape[0] < 1:
        raise ValueError("classes must not be empty.")
    prototypes = np.empty((class_values.shape[0], features.shape[1]), dtype=float)
    for index, class_label in enumerate(class_values.tolist()):
        mask = label_equal_mask(labels, class_label)
        if not np.any(mask):
            raise ValueError(f"No source rows available for class {class_label!r}.")
        prototypes[index] = np.mean(features[mask], axis=0)
    return prototypes


def prototype_distance_features(
    features: Sequence[Sequence[float]] | np.ndarray,
    prototypes: Sequence[Sequence[float]] | np.ndarray,
    *,
    metric: str = "squared_euclidean",
    output: str = "distance",
    feature_scale: Sequence[float] | np.ndarray | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Represent rows by distances or similarities to fixed prototypes."""

    matrix = _feature_matrix(features, name="features")
    proto = _feature_matrix(prototypes, name="prototypes")
    if matrix.shape[1] != proto.shape[1]:
        raise ValueError(f"features width {matrix.shape[1]} does not match prototype width {proto.shape[1]}.")
    metric_name = normalize_prototype_metric(metric)
    output_name = normalize_prototype_output(output)
    scale = np.ones(matrix.shape[1], dtype=float) if feature_scale is None else np.asarray(feature_scale, dtype=float).reshape(-1)
    if scale.shape[0] != matrix.shape[1] or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("feature_scale must contain one positive finite value per feature.")
    x = matrix / scale
    p = proto / scale
    if metric_name in {"squared_euclidean", "euclidean"}:
        diff = x[:, None, :] - p[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        if metric_name == "euclidean":
            distances = np.sqrt(np.maximum(distances, 0.0))
    else:
        x_norm = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), float(epsilon))
        p_norm = np.maximum(np.linalg.norm(p, axis=1, keepdims=True).T, float(epsilon))
        cosine = (x @ p.T) / (x_norm * p_norm)
        distances = 1.0 - np.clip(cosine, -1.0, 1.0)
    if output_name == "distance":
        return distances
    if output_name == "negative_distance":
        return -distances
    temp = _positive_float(temperature, name="temperature")
    return np.exp(-distances / temp)


def source_prototype_feature_config(
    *,
    metric: str | None = "squared_euclidean",
    output: str | None = "distance",
    use_diagonal_scale: bool | int | str = True,
    temperature: float | str = DEFAULT_TEMPERATURE,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourcePrototypeFeatureConfig:
    """Normalize source-prototype feature options."""

    return SourcePrototypeFeatureConfig(
        metric=normalize_prototype_metric(metric),
        output=normalize_prototype_output(output),
        use_diagonal_scale=_bool_value(use_diagonal_scale, name="use_diagonal_scale"),
        temperature=_positive_float(temperature, name="temperature"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_prototype_metric(value: str | None) -> str:
    """Normalize prototype-distance metric aliases."""

    normalized = "squared_euclidean" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"sqeuclidean": "squared_euclidean", "l2_squared": "squared_euclidean", "l2": "euclidean"}.get(normalized, normalized)
    if normalized not in PROTOTYPE_METRICS:
        raise ValueError(f"Unknown prototype metric {value!r}.")
    return normalized


def normalize_prototype_output(value: str | None) -> str:
    """Normalize prototype feature output aliases."""

    normalized = "distance" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"dist": "distance", "neg_distance": "negative_distance", "similarity": "rbf_similarity", "rbf": "rbf_similarity"}.get(normalized, normalized)
    if normalized not in PROTOTYPE_OUTPUTS:
        raise ValueError(f"Unknown prototype output {value!r}.")
    return normalized


def _coerce_config(config: SourcePrototypeFeatureConfig | Mapping[str, Any]) -> SourcePrototypeFeatureConfig:
    if isinstance(config, SourcePrototypeFeatureConfig):
        return config
    return source_prototype_feature_config(**dict(config))


def _metadata(cfg: SourcePrototypeFeatureConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, n_classes: int, counts: np.ndarray) -> dict[str, Any]:
    return {
        "source_prototype_features": True,
        "source_prototype_features_protocol": SOURCE_PROTOTYPE_FEATURES_PROTOCOL,
        "source_prototype_features_protocol_category": SOURCE_PROTOTYPE_FEATURES_CATEGORY,
        "source_prototype_features_metric": cfg.metric,
        "source_prototype_features_output": cfg.output,
        "source_prototype_features_uses_source_features": True,
        "source_prototype_features_uses_source_labels": True,
        "source_prototype_features_uses_test_features_for_fitting": False,
        "source_prototype_features_uses_test_labels": False,
        "source_prototype_features_valid_for_strict_source_only": True,
        "source_prototype_features_valid_for_benchmark": True,
        "source_prototype_features_n_source_rows": int(n_source_rows),
        "source_prototype_features_n_test_rows": int(n_test_rows),
        "source_prototype_features_input_dim": int(feature_dim),
        "source_prototype_features_n_classes": int(n_classes),
        "source_prototype_features_output_dim": int(n_classes),
        "source_prototype_features_use_diagonal_scale": bool(cfg.use_diagonal_scale),
        "source_prototype_features_temperature": float(cfg.temperature),
        "source_prototype_features_class_counts": "|".join(str(int(count)) for count in counts),
    }


def _feature_scale(features: np.ndarray, *, enabled: bool, epsilon: float) -> np.ndarray:
    if not enabled:
        return np.ones(features.shape[1], dtype=float)
    scale = np.std(features - np.mean(features, axis=0), axis=0, ddof=1 if features.shape[0] > 1 else 0)
    return np.maximum(scale, float(epsilon))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        vector = _object_vector([values])
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            vector = _object_vector([array.item()])
        elif array.ndim == 1 and array.shape[0] == expected_length:
            vector = array.reshape(-1)
        elif array.ndim == 1 and expected_length == 1:
            vector = _object_vector([tuple(array.tolist())])
        else:
            rows = array.reshape(array.shape[0], -1)
            vector = rows[:, 0].reshape(-1) if rows.shape[1] == 1 else _object_vector(tuple(row.tolist()) for row in rows)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _object_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


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
