"""Source-only Gaussian class-conditional decoder.

This module provides a dependency-light Protocol-1 baseline for cross-subject
feature decoding.  Class means, diagonal variances, and class priors are fitted
from source rows and source labels only.  Held-out rows are scored by Gaussian
log likelihoods and normalized to posterior probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask

SOURCE_GAUSSIAN_PROTOCOL = "strict_source_only_gaussian_decoder"
SOURCE_GAUSSIAN_CATEGORY = "1_strict_source_only"
COVARIANCE_TYPES = ("diagonal", "tied_diagonal", "spherical", "tied_spherical")
PRIOR_MODES = ("empirical", "uniform")
DEFAULT_VARIANCE_FLOOR = 1e-6
DEFAULT_TEMPERATURE = 1.0


@dataclass(frozen=True, slots=True)
class SourceGaussianConfig:
    """Configuration for the source-only Gaussian decoder."""

    covariance_type: str = "diagonal"
    prior: str = "empirical"
    variance_floor: float = DEFAULT_VARIANCE_FLOOR
    temperature: float = DEFAULT_TEMPERATURE


@dataclass(frozen=True, slots=True)
class SourceGaussianResult:
    """Gaussian decoder predictions and provenance metadata."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    priors: np.ndarray
    log_likelihoods: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_source_gaussian_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceGaussianConfig | Mapping[str, Any] | None = None,
) -> SourceGaussianResult:
    """Fit source Gaussian class models and score held-out rows.

    The fitting path uses only source features and source labels.  ``test_features``
    are used only for scoring.
    """

    cfg = source_gaussian_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    labels = _label_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    classes, _ = label_counts(labels)
    if classes.shape[0] < 2:
        raise ValueError("At least two source classes are required.")

    means, class_variances, counts = _class_gaussian_stats(source, labels, classes=classes, variance_floor=cfg.variance_floor)
    variances = _apply_covariance_type(class_variances, counts=counts, covariance_type=cfg.covariance_type, variance_floor=cfg.variance_floor)
    priors = _class_priors(counts, mode=cfg.prior)
    log_likelihoods = gaussian_log_likelihoods(test, means=means, variances=variances)
    log_posteriors = (log_likelihoods + np.log(priors)[None, :]) / cfg.temperature
    probabilities = _softmax(log_posteriors)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_test_rows=test.shape[0],
        feature_dim=source.shape[1],
        classes=classes,
        counts=counts,
    )
    return SourceGaussianResult(
        probabilities=probabilities.astype(np.float32, copy=False),
        predictions=predictions,
        classes=classes,
        means=means.astype(np.float32, copy=False),
        variances=variances.astype(np.float32, copy=False),
        priors=priors.astype(np.float32, copy=False),
        log_likelihoods=log_likelihoods.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_gaussian_config(
    *,
    covariance_type: str = "diagonal",
    prior: str = "empirical",
    variance_floor: float | str = DEFAULT_VARIANCE_FLOOR,
    temperature: float | str = DEFAULT_TEMPERATURE,
) -> SourceGaussianConfig:
    """Normalize public Gaussian decoder options."""

    return SourceGaussianConfig(
        covariance_type=normalize_covariance_type(covariance_type),
        prior=normalize_prior_mode(prior),
        variance_floor=_positive_float(variance_floor, name="variance_floor"),
        temperature=_positive_float(temperature, name="temperature"),
    )


def normalize_covariance_type(value: str | None) -> str:
    """Normalize covariance-type aliases."""

    normalized = "diagonal" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "diag": "diagonal",
        "class_diagonal": "diagonal",
        "shared_diagonal": "tied_diagonal",
        "tied_diag": "tied_diagonal",
        "class_spherical": "spherical",
        "shared_spherical": "tied_spherical",
        "tied_sphere": "tied_spherical",
    }.get(normalized, normalized)
    if normalized not in COVARIANCE_TYPES:
        raise ValueError(f"Unknown covariance_type {value!r}. Available values: {', '.join(COVARIANCE_TYPES)}.")
    return normalized


def normalize_prior_mode(value: str | None) -> str:
    """Normalize class-prior aliases."""

    normalized = "empirical" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"balanced": "uniform", "flat": "uniform", "frequency": "empirical", "counts": "empirical"}.get(normalized, normalized)
    if normalized not in PRIOR_MODES:
        raise ValueError(f"Unknown prior mode {value!r}. Available values: {', '.join(PRIOR_MODES)}.")
    return normalized


def gaussian_log_likelihoods(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    means: Sequence[Sequence[float]] | np.ndarray,
    variances: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return diagonal-Gaussian log likelihoods for every row/class pair."""

    x = _feature_matrix(features, name="features")
    mean_matrix = _feature_matrix(means, name="means")
    variance_matrix = _feature_matrix(variances, name="variances")
    if mean_matrix.shape != variance_matrix.shape:
        raise ValueError("means and variances must have the same shape.")
    if x.shape[1] != mean_matrix.shape[1]:
        raise ValueError(f"feature width {x.shape[1]} does not match mean width {mean_matrix.shape[1]}.")
    if np.any(variance_matrix <= 0.0):
        raise ValueError("variances must be positive.")
    diff = x[:, None, :] - mean_matrix[None, :, :]
    quadratic = np.sum((diff * diff) / variance_matrix[None, :, :], axis=2)
    log_det = np.sum(np.log(variance_matrix), axis=1)
    return -0.5 * (quadratic + log_det[None, :] + x.shape[1] * np.log(2.0 * np.pi))


def _coerce_config(config: SourceGaussianConfig | Mapping[str, Any]) -> SourceGaussianConfig:
    if isinstance(config, SourceGaussianConfig):
        return config
    return source_gaussian_config(**dict(config))


def _class_gaussian_stats(features: np.ndarray, labels: np.ndarray, *, classes: np.ndarray, variance_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros((classes.shape[0], features.shape[1]), dtype=float)
    variances = np.zeros_like(means)
    counts = np.zeros(classes.shape[0], dtype=int)
    for index, class_label in enumerate(classes.tolist()):
        mask = label_equal_mask(labels, class_label)
        counts[index] = int(np.count_nonzero(mask))
        if counts[index] == 0:
            raise ValueError(f"No source rows available for class {class_label!r}.")
        class_rows = features[mask]
        means[index] = np.mean(class_rows, axis=0)
        ddof = 1 if counts[index] > 1 else 0
        variances[index] = np.maximum(np.var(class_rows, axis=0, ddof=ddof), variance_floor)
    return means, variances, counts


def _apply_covariance_type(class_variances: np.ndarray, *, counts: np.ndarray, covariance_type: str, variance_floor: float) -> np.ndarray:
    if covariance_type == "diagonal":
        return class_variances
    if covariance_type == "spherical":
        spherical = np.mean(class_variances, axis=1, keepdims=True)
        return np.repeat(np.maximum(spherical, variance_floor), class_variances.shape[1], axis=1)
    weights = counts.astype(float) / float(np.sum(counts))
    tied = np.sum(class_variances * weights[:, None], axis=0, keepdims=True)
    if covariance_type == "tied_diagonal":
        return np.repeat(np.maximum(tied, variance_floor), class_variances.shape[0], axis=0)
    if covariance_type == "tied_spherical":
        shared = np.full((class_variances.shape[0], class_variances.shape[1]), max(float(np.mean(tied)), variance_floor), dtype=float)
        return shared
    raise ValueError(f"Unhandled covariance_type {covariance_type!r}.")


def _class_priors(counts: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.full(counts.shape[0], 1.0 / counts.shape[0], dtype=float)
    if mode == "empirical":
        return counts.astype(float) / float(np.sum(counts))
    raise ValueError(f"Unhandled prior mode {mode!r}.")


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _metadata(cfg: SourceGaussianConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, classes: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    return {
        "source_gaussian_decoder": True,
        "source_gaussian_protocol": SOURCE_GAUSSIAN_PROTOCOL,
        "source_gaussian_protocol_category": SOURCE_GAUSSIAN_CATEGORY,
        "source_gaussian_uses_source_features": True,
        "source_gaussian_uses_source_labels": True,
        "source_gaussian_uses_test_features_for_fitting": False,
        "source_gaussian_uses_test_labels": False,
        "source_gaussian_valid_for_strict_source_only": True,
        "source_gaussian_valid_for_benchmark": True,
        "source_gaussian_n_source_rows": int(n_source_rows),
        "source_gaussian_n_test_rows": int(n_test_rows),
        "source_gaussian_feature_dim": int(feature_dim),
        "source_gaussian_n_classes": int(classes.shape[0]),
        "source_gaussian_covariance_type": cfg.covariance_type,
        "source_gaussian_prior": cfg.prior,
        "source_gaussian_variance_floor": float(cfg.variance_floor),
        "source_gaussian_temperature": float(cfg.temperature),
        "source_gaussian_class_counts": "|".join(f"{label}:{int(count)}" for label, count in zip(classes.tolist(), counts, strict=True)),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        vector = _object_vector([values])
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            vector = _object_vector([array.item()])
        elif array.ndim == 1:
            if array.shape[0] == expected_length:
                vector = _object_vector(array.reshape(-1).tolist())
            elif expected_length == 1:
                vector = _object_vector([tuple(array.tolist())])
            else:
                vector = _object_vector(array.reshape(-1).tolist())
        else:
            rows = array.reshape(array.shape[0], -1)
            if rows.shape[1] == 1:
                vector = _object_vector(rows[:, 0].tolist())
            else:
                vector = _object_vector(tuple(row.tolist()) for row in rows)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
    return vector


def _object_vector(values: Sequence[Any]) -> np.ndarray:
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
