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
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


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
        projected = projected * _whitening_scale(reference) / denom
    return projected.astype(np.float32, copy=False)


def source_pca_config(
    *,
    n_components: int | str = DEFAULT_COMPONENTS,
    center: bool | str | int | float = True,
    scale: bool | str | int | float = False,
    whiten: bool | str | int | float = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourcePCAConfig:
    """Normalize public source-PCA options."""

    return SourcePCAConfig(
        n_components=_component_request(n_components),
        center=_bool_config(center, name="center"),
        scale=_bool_config(scale, name="scale"),
        whiten=_bool_config(whiten, name="whiten"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _coerce_config(config: SourcePCAConfig | Mapping[str, Any]) -> SourcePCAConfig:
    if isinstance(config, SourcePCAConfig):
        return source_pca_config(
            n_components=config.n_components,
            center=config.center,
            scale=config.scale,
            whiten=config.whiten,
            epsilon=config.epsilon,
        )
    return source_pca_config(**dict(config))


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a real bool while rejecting ambiguous truthy/falsy objects."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise _bool_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _bool_error(name)
        return _normalize_bool(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise _bool_error(name)
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        if np.isfinite(parsed) and parsed in {0.0, 1.0}:
            return bool(parsed)
        raise _bool_error(name)
    raise _bool_error(name)


def _resolve_components(value: int | str, *, n_rows: int, n_features: int, center: bool) -> int:
    maximum = min(n_features, max(1, n_rows - 1 if center and n_rows > 1 else n_rows))
    requested = _component_request(value)
    if isinstance(requested, str):
        return int(maximum)
    return min(int(requested), int(maximum))


def _whitening_scale(reference: SourcePCAReference) -> float:
    n_fit_rows = max(1, int(reference.n_fit_rows))
    variance_dof = n_fit_rows - 1 if reference.config.center and n_fit_rows > 1 else n_fit_rows
    return float(np.sqrt(float(max(1, variance_dof))))


def _component_request(value: Any) -> int | str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full"}:
            return text
        return _positive_integer(text, name="n_components")
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError("n_components must be a positive integer, 'all', or 'full'.")
        value = value.item()
    if isinstance(value, (bool, np.bool_, list, tuple, dict, set)):
        raise ValueError("n_components must be a positive integer, 'all', or 'full'.")
    return _positive_integer(value, name="n_components")


def _positive_integer(value: Any, *, name: str) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, 'all', or 'full'.") from None
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer, 'all', or 'full'.")
    return int(parsed)


def _bool_config(value: Any, *, name: str) -> bool:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a boolean value.")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
        if parsed in {0, 1}:
            return bool(parsed)
    if isinstance(value, (float, np.floating)):
        parsed_float = float(value)
        if np.isfinite(parsed_float) and parsed_float in {0.0, 1.0}:
            return bool(parsed_float)
    raise ValueError(f"{name} must be a boolean value.")


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


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be positive and finite.")
        value = value.item()
    if isinstance(value, (bool, np.bool_, list, tuple, dict, set)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be positive and finite.") from None
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
