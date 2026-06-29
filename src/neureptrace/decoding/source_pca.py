"""Strict source-only PCA projection helpers.

This module fits a PCA basis from source rows only and applies the fixed basis to
source and held-out feature matrices.  It is a Protocol-1 preprocessing helper:
held-out rows are transformed, but never used to fit the projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_PCA_PROTOCOL = "strict_source_only_pca_projection"
SOURCE_PCA_CATEGORY = "1_strict_source_only"
DEFAULT_COMPONENTS = 64
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourcePCAConfig:
    """Configuration for source-only PCA projection."""

    n_components: int | str = DEFAULT_COMPONENTS
    center: bool = True
    scale: bool = False
    whiten: bool = False
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourcePCAReference:
    """Fitted source-only PCA reference."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    config: SourcePCAConfig
    n_fit_rows: int


@dataclass(frozen=True, slots=True)
class SourcePCATransformResult:
    """Projected source/test rows and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourcePCAReference
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_pca_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourcePCAConfig | Mapping[str, Any] | None = None,
) -> SourcePCATransformResult:
    """Fit PCA on source rows and transform source/test rows."""

    cfg = source_pca_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    reference = fit_source_pca_reference(source, config=cfg)
    train = apply_source_pca_transform(source, reference)
    test_out = apply_source_pca_transform(test, reference)
    metadata = _metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], n_components=reference.components.shape[0])
    return SourcePCATransformResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        reference=reference,
        metadata=metadata,
    )


def fit_source_pca_reference(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourcePCAConfig | Mapping[str, Any] | None = None,
) -> SourcePCAReference:
    """Fit a PCA reference from source rows only."""

    cfg = source_pca_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    mean = np.mean(source, axis=0) if cfg.center else np.zeros(source.shape[1], dtype=float)
    centered = source - mean
    if cfg.scale:
        scale = np.std(centered, axis=0, ddof=1 if source.shape[0] > 1 else 0)
        scale = np.maximum(scale, cfg.epsilon)
    else:
        scale = np.ones(source.shape[1], dtype=float)
    prepared = centered / scale
    n_components = _resolve_components(cfg.n_components, n_rows=source.shape[0], n_features=source.shape[1], center=cfg.center)
    _u, singular_values, vt = np.linalg.svd(prepared, full_matrices=False)
    components = _canonicalize_component_signs(vt[:n_components])
    selected_singular_values = singular_values[:n_components]
    total_energy = float(np.sum(singular_values**2))
    if total_energy > 0.0:
        explained = (selected_singular_values**2) / total_energy
    else:
        explained = np.zeros(n_components, dtype=float)
    return SourcePCAReference(
        mean=mean.astype(float, copy=False),
        scale=scale.astype(float, copy=False),
        components=components.astype(np.float32, copy=False),
        singular_values=selected_singular_values.astype(float, copy=False),
        explained_variance_ratio=explained.astype(float, copy=False),
        config=cfg,
        n_fit_rows=int(source.shape[0]),
    )


def apply_source_pca_transform(features: Sequence[Sequence[float]] | np.ndarray, reference: SourcePCAReference) -> np.ndarray:
    """Project rows with a fitted source-only PCA reference."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != reference.components.shape[1]:
        raise ValueError(f"features width {matrix.shape[1]} does not match PCA reference width {reference.components.shape[1]}.")
    projected = ((matrix - reference.mean) / reference.scale) @ reference.components.T
    if reference.config.whiten:
        denom = np.maximum(reference.singular_values, reference.config.epsilon)
        projected = projected / denom
    return projected.astype(np.float32, copy=False)


def source_pca_config(
    *,
    n_components: int | str = DEFAULT_COMPONENTS,
    center: bool = True,
    scale: bool = False,
    whiten: bool = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourcePCAConfig:
    """Normalize public source-PCA options."""

    return SourcePCAConfig(
        n_components=n_components,
        center=bool(center),
        scale=bool(scale),
        whiten=bool(whiten),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _coerce_config(config: SourcePCAConfig | Mapping[str, Any]) -> SourcePCAConfig:
    if isinstance(config, SourcePCAConfig):
        return config
    return source_pca_config(**dict(config))


def _resolve_components(value: int | str, *, n_rows: int, n_features: int, center: bool) -> int:
    maximum = min(n_features, max(1, n_rows - 1 if center and n_rows > 1 else n_rows))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full"}:
            return int(maximum)
        requested = float(text)
    else:
        requested = float(value)
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1:
        raise ValueError("n_components must be a positive integer, 'all', or 'full'.")
    return min(int(requested), int(maximum))


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    output = np.asarray(components, dtype=float).copy()
    for row in range(output.shape[0]):
        pivot = int(np.argmax(np.abs(output[row])))
        if output[row, pivot] < 0.0:
            output[row] *= -1.0
    return output


def _metadata(cfg: SourcePCAConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, n_components: int) -> dict[str, Any]:
    return {
        "source_pca_transform": True,
        "source_pca_protocol": SOURCE_PCA_PROTOCOL,
        "source_pca_protocol_category": SOURCE_PCA_CATEGORY,
        "source_pca_uses_source_features": True,
        "source_pca_uses_test_features_for_fitting": False,
        "source_pca_uses_test_labels": False,
        "source_pca_valid_for_strict_source_only": True,
        "source_pca_valid_for_benchmark": True,
        "source_pca_n_source_rows": int(n_source_rows),
        "source_pca_n_test_rows": int(n_test_rows),
        "source_pca_feature_dim": int(feature_dim),
        "source_pca_n_components": int(n_components),
        "source_pca_requested_components": str(cfg.n_components),
        "source_pca_center": bool(cfg.center),
        "source_pca_scale": bool(cfg.scale),
        "source_pca_whiten": bool(cfg.whiten),
        "source_pca_epsilon": float(cfg.epsilon),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
