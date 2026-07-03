"""Strict source-only feature whitening and preconditioning.

This module estimates a fold-local feature transform from source rows only and
applies the frozen transform to source and held-out rows.  It is intended as a
small preprocessing baseline for cross-subject decoding when feature correlations
or scale differences can destabilize downstream decoders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_WHITENING_PROTOCOL = "strict_source_only_feature_whitening"
SOURCE_WHITENING_CATEGORY = "1_strict_source_only"
WHITENING_MODES = ("zca", "pca", "diagonal")
DEFAULT_REGULARIZATION = 1e-6
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceWhiteningConfig:
    """Configuration for source-fitted whitening."""

    mode: str = "zca"
    regularization: float = DEFAULT_REGULARIZATION
    center: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceWhiteningTransform:
    """Frozen source-fitted whitening transform."""

    mean: np.ndarray
    transform: np.ndarray
    eigenvalues: np.ndarray
    mode: str
    regularization: float
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceWhiteningResult:
    """Whitened source/test features and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    transform: SourceWhiteningTransform
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_whitening(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceWhiteningConfig | Mapping[str, Any] | None = None,
) -> SourceWhiteningResult:
    """Fit a source-only whitening transform and apply it to source/test rows.

    ``test_features`` are transformed with source-fitted statistics only.  They
    are not used to estimate the mean, covariance, or whitening matrix.
    """

    cfg = source_whitening_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    fitted = fit_source_whitening_transform(source, config=cfg)
    train = apply_source_whitening(source, fitted)
    test_out = apply_source_whitening(test, fitted)
    return SourceWhiteningResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        transform=fitted,
        metadata=_metadata(
            cfg,
            n_source_rows=source.shape[0],
            n_test_rows=test.shape[0],
            feature_dim=source.shape[1],
            output_dim=train.shape[1],
            eigenvalues=fitted.eigenvalues,
        ),
    )


def source_whitening_config(
    *,
    mode: str | None = "zca",
    regularization: float | str = DEFAULT_REGULARIZATION,
    center: bool | str | int | float = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceWhiteningConfig:
    """Normalize public whitening options."""

    return SourceWhiteningConfig(
        mode=normalize_whitening_mode(mode),
        regularization=_nonnegative_float(regularization, name="regularization"),
        center=_bool_config(center, name="center"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_whitening_mode(value: str | None) -> str:
    """Normalize whitening mode aliases."""

    normalized = "zca" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"zca_whitening": "zca", "pca_whitening": "pca", "diag": "diagonal", "standardize": "diagonal", "zscore": "diagonal"}.get(normalized, normalized)
    if normalized not in WHITENING_MODES:
        raise ValueError(f"Unknown whitening mode {value!r}. Available values: {', '.join(WHITENING_MODES)}.")
    return normalized


def fit_source_whitening_transform(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceWhiteningConfig | Mapping[str, Any] | None = None,
) -> SourceWhiteningTransform:
    """Estimate a whitening transform from source rows only."""

    cfg = source_whitening_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    mean = np.mean(source, axis=0) if cfg.center else np.zeros(source.shape[1], dtype=float)
    centered = source - mean
    if cfg.mode == "diagonal":
        variance = np.var(centered, axis=0, ddof=1 if source.shape[0] > 1 else 0)
        scale = 1.0 / np.sqrt(np.maximum(variance + cfg.regularization, cfg.epsilon))
        transform = np.diag(scale)
        eigenvalues = variance
    else:
        covariance = _covariance(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = _canonicalize_component_signs(eigenvectors[:, order])
        inv_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues + cfg.regularization, cfg.epsilon))
        if cfg.mode == "zca":
            transform = (eigenvectors * inv_sqrt) @ eigenvectors.T
        else:
            transform = eigenvectors * inv_sqrt
    return SourceWhiteningTransform(
        mean=mean.astype(float, copy=False),
        transform=transform.astype(float, copy=False),
        eigenvalues=eigenvalues.astype(float, copy=False),
        mode=cfg.mode,
        regularization=float(cfg.regularization),
        n_source_rows=int(source.shape[0]),
    )


def apply_source_whitening(features: Sequence[Sequence[float]] | np.ndarray, transform: SourceWhiteningTransform) -> np.ndarray:
    """Apply a frozen source-fitted whitening transform."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != transform.mean.shape[0] or matrix.shape[1] != transform.transform.shape[0]:
        raise ValueError("features width must match whitening transform.")
    return (matrix - transform.mean) @ transform.transform


def _coerce_config(config: SourceWhiteningConfig | Mapping[str, Any]) -> SourceWhiteningConfig:
    if isinstance(config, SourceWhiteningConfig):
        return config
    return source_whitening_config(**dict(config))


def _covariance(centered: np.ndarray) -> np.ndarray:
    if centered.shape[0] <= 1:
        return np.zeros((centered.shape[1], centered.shape[1]), dtype=float)
    return centered.T @ centered / float(centered.shape[0] - 1)


def _canonicalize_component_signs(eigenvectors: np.ndarray) -> np.ndarray:
    output = np.asarray(eigenvectors, dtype=float).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0.0:
            output[:, column] *= -1.0
    return output


def _metadata(
    cfg: SourceWhiteningConfig,
    *,
    n_source_rows: int,
    n_test_rows: int,
    feature_dim: int,
    output_dim: int,
    eigenvalues: np.ndarray,
) -> dict[str, Any]:
    return {
        "source_whitening": True,
        "source_whitening_protocol": SOURCE_WHITENING_PROTOCOL,
        "source_whitening_protocol_category": SOURCE_WHITENING_CATEGORY,
        "source_whitening_mode": cfg.mode,
        "source_whitening_uses_source_features": True,
        "source_whitening_uses_test_features_for_fitting": False,
        "source_whitening_uses_test_labels": False,
        "source_whitening_valid_for_strict_source_only": True,
        "source_whitening_valid_for_benchmark": True,
        "source_whitening_n_source_rows": int(n_source_rows),
        "source_whitening_n_test_rows": int(n_test_rows),
        "source_whitening_feature_dim": int(feature_dim),
        "source_whitening_output_dim": int(output_dim),
        "source_whitening_regularization": float(cfg.regularization),
        "source_whitening_center": bool(cfg.center),
        "source_whitening_epsilon": float(cfg.epsilon),
        "source_whitening_min_eigenvalue": float(np.min(eigenvalues)) if eigenvalues.size else 0.0,
        "source_whitening_max_eigenvalue": float(np.max(eigenvalues)) if eigenvalues.size else 0.0,
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _bool_config(value: bool | str | int | float, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean value.")


def _positive_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be non-negative and finite.") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed
