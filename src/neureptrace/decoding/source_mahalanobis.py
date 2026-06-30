"""Strict source-only Mahalanobis decoder.

This module provides a dependency-light nearest-class decoder with a regularized
tied covariance estimate.  Class means and covariance are fitted from source rows
and source labels only.  Rows to score are evaluated by negative Mahalanobis
distance and converted to probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_MAHALANOBIS_PROTOCOL = "strict_source_only_mahalanobis_decoder"
SOURCE_MAHALANOBIS_CATEGORY = "1_strict_source_only"
PRIOR_MODES = ("empirical", "uniform")
DEFAULT_REGULARIZATION = 1e-3
DEFAULT_TEMPERATURE = 1.0


@dataclass(frozen=True, slots=True)
class SourceMahalanobisConfig:
    """Configuration for the source-only Mahalanobis decoder."""

    regularization: float = DEFAULT_REGULARIZATION
    prior: str = "empirical"
    temperature: float = DEFAULT_TEMPERATURE


@dataclass(frozen=True, slots=True)
class SourceMahalanobisResult:
    """Mahalanobis decoder predictions and provenance metadata."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    means: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    priors: np.ndarray
    distances: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-locals

def fit_source_mahalanobis_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceMahalanobisConfig | Mapping[str, Any] | None = None,
) -> SourceMahalanobisResult:
    """Fit source class means/tied covariance and score rows."""

    cfg = source_mahalanobis_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    labels = _label_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    classes = _unique_labels(labels)
    if classes.shape[0] < 2:
        raise ValueError("At least two source classes are required.")

    means, counts = _class_means(source, labels, classes=classes)
    covariance = tied_covariance(source, labels, classes=classes, means=means, regularization=cfg.regularization)
    precision = np.linalg.pinv(covariance)
    distances = mahalanobis_distances(test, means=means, precision=precision)
    priors = _class_priors(counts, prior=cfg.prior)
    scores = -distances / cfg.temperature + np.log(priors)[None, :]
    probabilities = _softmax(scores)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metadata = _metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], classes=classes, counts=counts)
    return SourceMahalanobisResult(
        probabilities=probabilities.astype(np.float32, copy=False),
        predictions=predictions,
        classes=classes,
        means=means.astype(np.float32, copy=False),
        covariance=covariance.astype(np.float32, copy=False),
        precision=precision.astype(np.float32, copy=False),
        priors=priors.astype(np.float32, copy=False),
        distances=distances.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_mahalanobis_config(
    *,
    regularization: float | str = DEFAULT_REGULARIZATION,
    prior: str | None = "empirical",
    temperature: float | str = DEFAULT_TEMPERATURE,
) -> SourceMahalanobisConfig:
    """Normalize public Mahalanobis decoder options."""

    return SourceMahalanobisConfig(
        regularization=_nonnegative_float(regularization, name="regularization"),
        prior=normalize_prior_mode(prior),
        temperature=_positive_float(temperature, name="temperature"),
    )


def normalize_prior_mode(value: str | None) -> str:
    """Normalize class-prior aliases."""

    normalized = "empirical" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"balanced": "uniform", "flat": "uniform", "frequency": "empirical", "counts": "empirical"}.get(normalized, normalized)
    if normalized not in PRIOR_MODES:
        raise ValueError(f"Unknown prior mode {value!r}. Available values: {', '.join(PRIOR_MODES)}.")
    return normalized


def tied_covariance(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    means: Sequence[Sequence[float]] | np.ndarray | None = None,
    regularization: float = DEFAULT_REGULARIZATION,
) -> np.ndarray:
    """Estimate a regularized tied within-class covariance matrix."""

    x = _feature_matrix(features, name="features")
    y = _label_vector(labels, expected_length=x.shape[0], name="labels")
    class_values = _unique_labels(y) if classes is None else _coerce_label_values(classes)
    mean_matrix = np.asarray(means, dtype=float) if means is not None else _class_means(x, y, classes=class_values)[0]
    if mean_matrix.shape != (class_values.shape[0], x.shape[1]):
        raise ValueError("means must have shape n_classes x n_features.")
    scatter = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    total = 0
    for index, class_label in enumerate(class_values.tolist()):
        rows = x[_label_equal_mask(y, class_label)]
        if rows.size == 0:
            continue
        centered = rows - mean_matrix[index]
        scatter += centered.T @ centered
        total += rows.shape[0]
    denominator = max(1, total - class_values.shape[0])
    covariance = scatter / float(denominator)
    covariance += _nonnegative_float(regularization, name="regularization") * np.eye(x.shape[1], dtype=float)
    return _nearest_spd(covariance)


def mahalanobis_distances(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    means: Sequence[Sequence[float]] | np.ndarray,
    precision: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return squared Mahalanobis distances to class means."""

    x = _feature_matrix(features, name="features")
    mean_matrix = _feature_matrix(means, name="means")
    precision_matrix = np.asarray(precision, dtype=float)
    if precision_matrix.shape != (x.shape[1], x.shape[1]):
        raise ValueError("precision must be square with feature width.")
    if mean_matrix.shape[1] != x.shape[1]:
        raise ValueError("means width must match feature width.")
    diff = x[:, None, :] - mean_matrix[None, :, :]
    return np.einsum("ncf,fg,ncg->nc", diff, precision_matrix, diff)


def _coerce_config(config: SourceMahalanobisConfig | Mapping[str, Any]) -> SourceMahalanobisConfig:
    if isinstance(config, SourceMahalanobisConfig):
        return config
    return source_mahalanobis_config(**dict(config))


def _class_means(features: np.ndarray, labels: np.ndarray, *, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((classes.shape[0], features.shape[1]), dtype=float)
    counts = np.zeros(classes.shape[0], dtype=int)
    for index, class_label in enumerate(classes.tolist()):
        mask = _label_equal_mask(labels, class_label)
        counts[index] = int(np.count_nonzero(mask))
        if counts[index] == 0:
            raise ValueError(f"No source rows available for class {class_label!r}.")
        means[index] = np.mean(features[mask], axis=0)
    return means, counts


def _class_priors(counts: np.ndarray, *, prior: str) -> np.ndarray:
    if prior == "uniform":
        return np.full(counts.shape[0], 1.0 / counts.shape[0], dtype=float)
    if prior == "empirical":
        return counts.astype(float) / float(np.sum(counts))
    raise ValueError(f"Unhandled prior mode {prior!r}.")


def _nearest_spd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _metadata(cfg: SourceMahalanobisConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, classes: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    return {
        "source_mahalanobis_decoder": True,
        "source_mahalanobis_protocol": SOURCE_MAHALANOBIS_PROTOCOL,
        "source_mahalanobis_protocol_category": SOURCE_MAHALANOBIS_CATEGORY,
        "source_mahalanobis_uses_source_features": True,
        "source_mahalanobis_uses_source_labels": True,
        "source_mahalanobis_uses_test_features_for_fitting": False,
        "source_mahalanobis_uses_test_labels": False,
        "source_mahalanobis_valid_for_strict_source_only": True,
        "source_mahalanobis_valid_for_benchmark": True,
        "source_mahalanobis_n_source_rows": int(n_source_rows),
        "source_mahalanobis_n_test_rows": int(n_test_rows),
        "source_mahalanobis_feature_dim": int(feature_dim),
        "source_mahalanobis_n_classes": int(classes.shape[0]),
        "source_mahalanobis_regularization": float(cfg.regularization),
        "source_mahalanobis_prior": cfg.prior,
        "source_mahalanobis_temperature": float(cfg.temperature),
        "source_mahalanobis_class_counts": "|".join(f"{label}:{int(count)}" for label, count in zip(classes.tolist(), counts, strict=True)),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = _coerce_label_values(values)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
    return vector


def _coerce_label_values(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            items = array.tolist()
        elif array.shape[1:] == (1,):
            items = array.reshape(-1).tolist()
        else:
            items = [tuple(np.asarray(row, dtype=object).reshape(-1).tolist()) for row in array]
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
    return _object_vector(_canonical_label(item) for item in items)


def _canonical_label(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return array.item()
        return tuple(array.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    vector[:] = items
    return vector


def _unique_labels(labels: np.ndarray) -> np.ndarray:
    classes: list[Any] = []
    for label in labels.tolist():
        if not any(_labels_equal(label, class_label) for class_label in classes):
            classes.append(label)
    return _object_vector(classes)


def _label_equal_mask(labels: np.ndarray, class_label: Any) -> np.ndarray:
    return np.asarray([_labels_equal(label, class_label) for label in labels.tolist()], dtype=bool)


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


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
