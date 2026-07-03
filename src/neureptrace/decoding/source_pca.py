"""Strict source-only PCA projection.

This module estimates a principal-component projection from source rows only and
then applies that source-fitted projection to source and held-out rows.  It is a
small fold-local preprocessing baseline for cross-subject decoding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_PCA_PROTOCOL = "strict_source_only_pca_projection"
SOURCE_PCA_CATEGORY = "1_strict_source_only"
DEFAULT_COMPONENTS = "all"
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourcePCAConfig:
    """Configuration for source-fitted PCA projection."""

    n_components: int | str | float = DEFAULT_COMPONENTS
    center: bool = True
    scale: bool = False
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourcePCAProjector:
    """Source-fitted PCA projection."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    n_source_rows: int
    feature_dim: int


@dataclass(frozen=True, slots=True)
class SourcePCAResult:
    """Projected source/test features and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    projector: SourcePCAProjector
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_pca_projection(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourcePCAConfig | Mapping[str, Any] | None = None,
) -> SourcePCAResult:
    """Fit PCA on source rows and transform source/test matrices."""

    cfg = source_pca_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    projector = fit_source_pca_projector(source, config=cfg)
    train_projected = transform_with_source_pca(source, projector)
    test_projected = transform_with_source_pca(test, projector)
    return SourcePCAResult(
        train_features=train_projected.astype(np.float32, copy=False),
        test_features=test_projected.astype(np.float32, copy=False),
        projector=projector,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], output_dim=train_projected.shape[1]),
    )


def source_pca_config(
    *,
    n_components: int | str | float = DEFAULT_COMPONENTS,
    center: bool | str | int | float = True,
    scale: bool | str | int | float = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourcePCAConfig:
    """Normalize public PCA options."""

    return SourcePCAConfig(
        n_components=n_components,
        center=_bool_config(center, name="center"),
        scale=_bool_config(scale, name="scale"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def fit_source_pca_projector(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourcePCAConfig | Mapping[str, Any] | None = None,
) -> SourcePCAProjector:
    """Estimate a PCA projector from source rows only."""

    cfg = source_pca_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    n_components = _effective_components(cfg.n_components, max_components=min(source.shape[0], source.shape[1]))
    mean = np.mean(source, axis=0) if cfg.center else np.zeros(source.shape[1], dtype=float)
    centered = source - mean
    if cfg.scale:
        scale = np.std(centered, axis=0, ddof=1 if source.shape[0] > 1 else 0)
        scale = np.maximum(scale, cfg.epsilon)
    else:
        scale = np.ones(source.shape[1], dtype=float)
    prepared = centered / scale
    _u, singular_values, vt = np.linalg.svd(prepared, full_matrices=False)
    components = _canonicalize_rows(vt[:n_components])
    selected = singular_values[:n_components]
    total_energy = float(np.sum(singular_values**2))
    explained = np.zeros(n_components, dtype=float) if total_energy <= 0.0 else (selected**2) / total_energy
    return SourcePCAProjector(
        mean=mean.astype(float, copy=False),
        scale=scale.astype(float, copy=False),
        components=components.astype(np.float32, copy=False),
        singular_values=selected.astype(float, copy=False),
        explained_variance_ratio=explained.astype(float, copy=False),
        n_source_rows=int(source.shape[0]),
        feature_dim=int(source.shape[1]),
    )


def transform_with_source_pca(features: Sequence[Sequence[float]] | np.ndarray, projector: SourcePCAProjector) -> np.ndarray:
    """Apply a source-fitted PCA projector."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != projector.feature_dim:
        raise ValueError(f"features width {matrix.shape[1]} does not match projector width {projector.feature_dim}.")
    return ((matrix - projector.mean) / projector.scale) @ projector.components.T


def reconstruct_from_source_pca(scores: Sequence[Sequence[float]] | np.ndarray, projector: SourcePCAProjector) -> np.ndarray:
    """Map PCA scores back to the original feature space."""

    matrix = _feature_matrix(scores, name="scores")
    if matrix.shape[1] != projector.components.shape[0]:
        raise ValueError("scores width must match the number of PCA components.")
    return (matrix @ projector.components) * projector.scale + projector.mean


def _coerce_config(config: SourcePCAConfig | Mapping[str, Any]) -> SourcePCAConfig:
    if isinstance(config, SourcePCAConfig):
        return config
    return source_pca_config(**dict(config))


def _effective_components(value: int | str | float, *, max_components: int) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "all", "full", "inf", "infinity"}:
            return int(max_components)
        requested = float(text)
    else:
        requested = float(value)
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1.0:
        raise ValueError("n_components must be a positive integer or 'all'.")
    return min(int(requested), int(max_components))


def _canonicalize_rows(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=float).copy()
    for row in range(output.shape[0]):
        pivot = int(np.argmax(np.abs(output[row])))
        if output[row, pivot] < 0.0:
            output[row] *= -1.0
    return output


def _metadata(cfg: SourcePCAConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, output_dim: int) -> dict[str, Any]:
    return {
        "source_pca_projection": True,
        "source_pca_protocol": SOURCE_PCA_PROTOCOL,
        "source_pca_protocol_category": SOURCE_PCA_CATEGORY,
        "source_pca_uses_source_features": True,
        "source_pca_uses_test_features_for_fitting": False,
        "source_pca_uses_test_labels": False,
        "source_pca_valid_for_strict_source_only": True,
        "source_pca_valid_for_benchmark": True,
        "source_pca_requested_components": str(cfg.n_components),
        "source_pca_center": bool(cfg.center),
        "source_pca_scale": bool(cfg.scale),
        "source_pca_epsilon": float(cfg.epsilon),
        "source_pca_n_source_rows": int(n_source_rows),
        "source_pca_n_test_rows": int(n_test_rows),
        "source_pca_feature_dim": int(feature_dim),
        "source_pca_output_dim": int(output_dim),
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
