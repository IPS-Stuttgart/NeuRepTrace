"""Strict source-only PCA projection.

This module fits a PCA basis from source rows only and applies the fitted
projection to source and held-out rows.  The held-out rows are transformed but are
not used to estimate means, scales, components, singular values, or whitening
factors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_PCA_PROTOCOL = "strict_source_only_pca_projection"
SOURCE_PCA_CATEGORY = "1_strict_source_only"
DEFAULT_COMPONENTS = 64
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourcePCAConfig:
    """Configuration for source-only PCA."""

    n_components: int | str | float = DEFAULT_COMPONENTS
    center: bool = True
    scale: bool = False
    whiten: bool = False
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourcePCAProjection:
    """PCA projection fitted from source rows only."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    n_source_rows: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourcePCAResult:
    """Source/test features transformed with a source-fitted PCA projection."""

    train_features: np.ndarray
    test_features: np.ndarray
    projection: SourcePCAProjection
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_pca(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourcePCAConfig | Mapping[str, Any] | None = None,
) -> SourcePCAResult:
    """Fit PCA on source rows and transform source/test rows.

    ``test_features`` are used only for transformation, never for fitting.
    """

    cfg = source_pca_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    projection = fit_source_pca_projection(source, config=cfg)
    train = transform_with_source_pca(source, projection, whiten=cfg.whiten, epsilon=cfg.epsilon)
    transformed = transform_with_source_pca(test, projection, whiten=cfg.whiten, epsilon=cfg.epsilon)
    metadata = {
        **projection.metadata,
        "source_pca_n_test_rows": int(test.shape[0]),
        "source_pca_train_output_dim": int(train.shape[1]),
        "source_pca_test_output_dim": int(transformed.shape[1]),
    }
    return SourcePCAResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=transformed.astype(np.float32, copy=False),
        projection=projection,
        metadata=metadata,
    )


def source_pca_config(
    *,
    n_components: int | str | float = DEFAULT_COMPONENTS,
    center: bool | str | int | float = True,
    scale: bool | str | int | float = False,
    whiten: bool | str | int | float = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourcePCAConfig:
    """Normalize public PCA options."""

    return SourcePCAConfig(
        n_components=n_components,
        center=_bool_config(center, name="center"),
        scale=_bool_config(scale, name="scale"),
        whiten=_bool_config(whiten, name="whiten"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def fit_source_pca_projection(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourcePCAConfig | Mapping[str, Any] | None = None,
) -> SourcePCAProjection:
    """Fit a PCA projection from source rows only."""

    cfg = source_pca_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    prepared, mean, scale_vector = _prepare_source(source, center=cfg.center, scale=cfg.scale, epsilon=cfg.epsilon)
    max_components = min(prepared.shape[0], prepared.shape[1])
    if cfg.center and prepared.shape[0] > 1:
        max_components = min(max_components, prepared.shape[0] - 1)
    max_components = max(1, max_components)
    n_components = _effective_components(cfg.n_components, max_components=max_components)
    _u, singular_values, vt = np.linalg.svd(prepared, full_matrices=False)
    components = _canonicalize_signs(vt[:n_components])
    selected_singular = singular_values[:n_components]
    total_energy = float(np.sum(singular_values**2))
    explained = np.zeros(n_components, dtype=float) if total_energy <= 0.0 else (selected_singular**2) / total_energy
    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        feature_dim=source.shape[1],
        n_components=n_components,
        explained=explained,
    )
    return SourcePCAProjection(
        mean=mean.astype(float, copy=False),
        scale=scale_vector.astype(float, copy=False),
        components=components.astype(np.float32, copy=False),
        singular_values=selected_singular.astype(float, copy=False),
        explained_variance_ratio=explained.astype(float, copy=False),
        n_source_rows=int(source.shape[0]),
        metadata=metadata,
    )


def transform_with_source_pca(
    features: Sequence[Sequence[float]] | np.ndarray,
    projection: SourcePCAProjection,
    *,
    whiten: bool | str | int | float = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> np.ndarray:
    """Transform rows with a source-fitted PCA projection."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != projection.components.shape[1]:
        raise ValueError(
            "features width does not match PCA projection width: "
            f"{matrix.shape[1]} != {projection.components.shape[1]}."
        )
    prepared = (matrix - projection.mean) / projection.scale
    transformed = prepared @ projection.components.T
    if _bool_config(whiten, name="whiten"):
        denom = np.maximum(projection.singular_values, _positive_float(epsilon, name="epsilon"))
        transformed = transformed * np.sqrt(max(projection.n_source_rows - 1, 1)) / denom
    return transformed.astype(np.float32, copy=False)


def _coerce_config(config: SourcePCAConfig | Mapping[str, Any]) -> SourcePCAConfig:
    if isinstance(config, SourcePCAConfig):
        return config
    return source_pca_config(**dict(config))


def _prepare_source(source: np.ndarray, *, center: bool, scale: bool, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(source, axis=0) if center else np.zeros(source.shape[1], dtype=float)
    centered = source - mean
    if scale:
        scale_vector = np.std(centered, axis=0, ddof=1 if source.shape[0] > 1 else 0)
        scale_vector = np.maximum(scale_vector, float(epsilon))
    else:
        scale_vector = np.ones(source.shape[1], dtype=float)
    return centered / scale_vector, mean, scale_vector


def _effective_components(value: int | str | float, *, max_components: int) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            requested = DEFAULT_COMPONENTS
        elif text in {"all", "full", "inf", "infinity"}:
            return int(max_components)
        else:
            requested = float(text)
    else:
        requested = float(value)
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1.0:
        raise ValueError("n_components must be a positive integer, 'all', or infinity.")
    return min(int(requested), int(max_components))


def _canonicalize_signs(components: np.ndarray) -> np.ndarray:
    output = np.asarray(components, dtype=float).copy()
    for row in range(output.shape[0]):
        pivot = int(np.argmax(np.abs(output[row])))
        if output[row, pivot] < 0.0:
            output[row] *= -1.0
    return output


def _metadata(cfg: SourcePCAConfig, *, n_source_rows: int, feature_dim: int, n_components: int, explained: np.ndarray) -> dict[str, Any]:
    return {
        "source_pca": True,
        "source_pca_protocol": SOURCE_PCA_PROTOCOL,
        "source_pca_protocol_category": SOURCE_PCA_CATEGORY,
        "source_pca_uses_source_features": True,
        "source_pca_uses_test_features_for_fitting": False,
        "source_pca_uses_test_labels": False,
        "source_pca_valid_for_strict_source_only": True,
        "source_pca_valid_for_benchmark": True,
        "source_pca_n_source_rows": int(n_source_rows),
        "source_pca_feature_dim": int(feature_dim),
        "source_pca_n_components": int(n_components),
        "source_pca_requested_components": str(cfg.n_components),
        "source_pca_center": bool(cfg.center),
        "source_pca_scale": bool(cfg.scale),
        "source_pca_whiten": bool(cfg.whiten),
        "source_pca_explained_variance_sum": float(np.sum(explained)),
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
