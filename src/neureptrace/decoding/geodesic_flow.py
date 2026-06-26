"""Sampled geodesic-flow features for Category-2 transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GEODESIC_FLOW_PROTOCOL = "unlabeled_target_sampled_geodesic_flow"
GEODESIC_FLOW_CATEGORY = "2_unlabeled_target_adaptive"
DEFAULT_GEODESIC_COMPONENTS = 16
DEFAULT_GEODESIC_STEPS = 5
TARGET_FEATURE_SOURCE_TRANSDUCTIVE = "target_test_features_transductive"
TARGET_FEATURE_SOURCE_CALIBRATION = "target_adaptation_features"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


@dataclass(frozen=True, slots=True)
class GeodesicFlowConfig:
    """Configuration for sampled geodesic-flow features."""

    n_components: int | str | float = DEFAULT_GEODESIC_COMPONENTS
    n_steps: int | str = DEFAULT_GEODESIC_STEPS
    center: bool = True
    scale: bool = False
    include_endpoints: bool = True
    normalize_blocks: bool = True
    epsilon: float = 1e-8


@dataclass(frozen=True, slots=True)
class GeodesicFlowBasis:
    """One sampled orthonormal basis along the source-target path."""

    position: float
    basis: np.ndarray


@dataclass(frozen=True, slots=True)
class GeodesicFlowResult:
    """Geodesic-flow train/test features and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    source_components: np.ndarray
    target_components: np.ndarray
    bases: tuple[GeodesicFlowBasis, ...]
    source_mean: np.ndarray
    target_mean: np.ndarray
    source_scale: np.ndarray
    target_scale: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_sampled_geodesic_flow_features(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_test_features: Sequence[Sequence[float]] | np.ndarray,
    config: GeodesicFlowConfig | Mapping[str, Any] | None = None,
    target_adaptation_features: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> GeodesicFlowResult:
    """Fit sampled geodesic-flow features from source and unlabeled target data."""

    cfg = geodesic_flow_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    target_test = _feature_matrix(target_test_features, name="target_test_features")
    if source.shape[1] != target_test.shape[1]:
        raise ValueError(
            "source_features and target_test_features must have the same feature width: "
            f"{source.shape[1]} != {target_test.shape[1]}."
        )
    target_fit = target_test if target_adaptation_features is None else _feature_matrix(target_adaptation_features, name="target_adaptation_features")
    if target_fit.shape[1] != source.shape[1]:
        raise ValueError(
            "target_adaptation_features and source_features must have the same feature width: "
            f"{target_fit.shape[1]} != {source.shape[1]}."
        )

    n_components = _effective_components(cfg.n_components, max_components=_max_components(source, target_fit, center=cfg.center))
    n_steps = _positive_int(cfg.n_steps, name="n_steps")
    source_prepared, source_mean, source_scale = _prepare_domain(source, center=cfg.center, scale=cfg.scale, epsilon=cfg.epsilon)
    target_fit_prepared, target_mean, target_scale = _prepare_domain(target_fit, center=cfg.center, scale=cfg.scale, epsilon=cfg.epsilon)
    target_test_prepared = (target_test - target_mean) / target_scale
    source_components = _pca_components(source_prepared, n_components=n_components)
    target_components = _pca_components(target_fit_prepared, n_components=n_components)
    bases = sample_geodesic_bases(source_components, target_components, n_steps=n_steps, include_endpoints=cfg.include_endpoints)
    train_features = transform_with_geodesic_bases(source_prepared, bases, normalize_blocks=cfg.normalize_blocks)
    test_features = transform_with_geodesic_bases(target_test_prepared, bases, normalize_blocks=cfg.normalize_blocks)

    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_target_fit_rows=target_fit.shape[0],
        n_target_test_rows=target_test.shape[0],
        feature_dim=source.shape[1],
        n_components=n_components,
        n_bases=len(bases),
        target_feature_source=TARGET_FEATURE_SOURCE_TRANSDUCTIVE if target_adaptation_features is None else TARGET_FEATURE_SOURCE_CALIBRATION,
    )
    return GeodesicFlowResult(
        train_features=train_features.astype(np.float32, copy=False),
        test_features=test_features.astype(np.float32, copy=False),
        source_components=source_components.astype(np.float32, copy=False),
        target_components=target_components.astype(np.float32, copy=False),
        bases=bases,
        source_mean=source_mean.astype(float, copy=False),
        target_mean=target_mean.astype(float, copy=False),
        source_scale=source_scale.astype(float, copy=False),
        target_scale=target_scale.astype(float, copy=False),
        metadata=metadata,
    )


def geodesic_flow_config(
    *,
    n_components: int | str | float = DEFAULT_GEODESIC_COMPONENTS,
    n_steps: int | str = DEFAULT_GEODESIC_STEPS,
    center: bool | str | int = True,
    scale: bool | str | int = False,
    include_endpoints: bool | str | int = True,
    normalize_blocks: bool | str | int = True,
    epsilon: float | str = 1e-8,
) -> GeodesicFlowConfig:
    """Normalize public sampled-geodesic options."""

    return GeodesicFlowConfig(
        n_components=n_components,
        n_steps=_positive_int(n_steps, name="n_steps"),
        center=_boolean_option(center, name="center"),
        scale=_boolean_option(scale, name="scale"),
        include_endpoints=_boolean_option(include_endpoints, name="include_endpoints"),
        normalize_blocks=_boolean_option(normalize_blocks, name="normalize_blocks"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def sample_geodesic_bases(
    source_components: Sequence[Sequence[float]] | np.ndarray,
    target_components: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_steps: int | str = DEFAULT_GEODESIC_STEPS,
    include_endpoints: bool | str | int = True,
) -> tuple[GeodesicFlowBasis, ...]:
    """Sample orthonormal bases between source and target PCA bases."""

    source = _component_matrix(source_components, name="source_components")
    target = _component_matrix(target_components, name="target_components")
    if source.shape != target.shape:
        raise ValueError(f"source_components and target_components must have the same shape: {source.shape} != {target.shape}.")
    steps = _positive_int(n_steps, name="n_steps")
    endpoints = _boolean_option(include_endpoints, name="include_endpoints")
    positions = np.linspace(0.0, 1.0, steps if endpoints else steps + 2)
    if not endpoints:
        positions = positions[1:-1]
    bases = []
    for position in positions:
        interpolated = (1.0 - float(position)) * source.T + float(position) * target.T
        q, _r = np.linalg.qr(interpolated)
        basis = _canonicalize_component_signs(q[:, : source.shape[0]].T)
        bases.append(GeodesicFlowBasis(position=float(position), basis=basis.astype(np.float32, copy=False)))
    return tuple(bases)


def transform_with_geodesic_bases(
    features: Sequence[Sequence[float]] | np.ndarray,
    bases: Sequence[GeodesicFlowBasis],
    *,
    normalize_blocks: bool | str | int = True,
) -> np.ndarray:
    """Concatenate projections onto sampled geodesic bases."""

    matrix = _feature_matrix(features, name="features")
    basis_tuple = tuple(bases)
    if not basis_tuple:
        raise ValueError("At least one geodesic basis is required.")
    normalize = _boolean_option(normalize_blocks, name="normalize_blocks")
    blocks = []
    for basis in basis_tuple:
        basis_matrix = _component_matrix(basis.basis, name="basis")
        if basis_matrix.shape[1] != matrix.shape[1]:
            raise ValueError(f"basis width {basis_matrix.shape[1]} does not match feature width {matrix.shape[1]}.")
        block = matrix @ basis_matrix.T
        if normalize:
            block = block / np.sqrt(len(basis_tuple))
        blocks.append(block)
    return np.hstack(blocks).astype(np.float32, copy=False)


def _coerce_config(config: GeodesicFlowConfig | Mapping[str, Any]) -> GeodesicFlowConfig:
    if isinstance(config, GeodesicFlowConfig):
        return geodesic_flow_config(
            n_components=config.n_components,
            n_steps=config.n_steps,
            center=config.center,
            scale=config.scale,
            include_endpoints=config.include_endpoints,
            normalize_blocks=config.normalize_blocks,
            epsilon=config.epsilon,
        )
    return geodesic_flow_config(**dict(config))


def _boolean_option(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise ValueError(f"{name} must be a boolean value.")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _boolean_option(value.item(), name=name)
        raise ValueError(f"{name} must be a boolean value.")
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean value.")


def _prepare_domain(matrix: np.ndarray, *, center: bool, scale: bool, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0) if center else np.zeros(matrix.shape[1], dtype=float)
    centered = matrix - mean
    if scale:
        variance = np.var(centered, axis=0, ddof=1 if matrix.shape[0] > 1 else 0)
        scale_vector = np.sqrt(np.maximum(variance, _positive_float(epsilon, name="epsilon")))
    else:
        scale_vector = np.ones(matrix.shape[1], dtype=float)
    return centered / scale_vector, mean, scale_vector


def _pca_components(matrix: np.ndarray, *, n_components: int) -> np.ndarray:
    _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
    return _canonicalize_component_signs(vt[:n_components])


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    output = np.asarray(components, dtype=float).copy()
    for row in range(output.shape[0]):
        pivot = int(np.argmax(np.abs(output[row])))
        if output[row, pivot] < 0.0:
            output[row] *= -1.0
    return output


def _metadata(
    cfg: GeodesicFlowConfig,
    *,
    n_source_rows: int,
    n_target_fit_rows: int,
    n_target_test_rows: int,
    feature_dim: int,
    n_components: int,
    n_bases: int,
    target_feature_source: str,
) -> dict[str, Any]:
    return {
        "geodesic_flow": True,
        "geodesic_flow_protocol": GEODESIC_FLOW_PROTOCOL,
        "geodesic_flow_protocol_category": GEODESIC_FLOW_CATEGORY,
        "geodesic_flow_uses_source_features": True,
        "geodesic_flow_uses_source_labels": False,
        "geodesic_flow_uses_target_features": True,
        "geodesic_flow_uses_target_labels": False,
        "geodesic_flow_valid_for_strict_source_only": False,
        "geodesic_flow_valid_for_unlabeled_target_adaptation": True,
        "geodesic_flow_valid_for_benchmark": False,
        "geodesic_flow_target_feature_source": target_feature_source,
        "geodesic_flow_transductive": target_feature_source == TARGET_FEATURE_SOURCE_TRANSDUCTIVE,
        "geodesic_flow_n_source_rows": int(n_source_rows),
        "geodesic_flow_n_target_fit_rows": int(n_target_fit_rows),
        "geodesic_flow_n_target_test_rows": int(n_target_test_rows),
        "geodesic_flow_feature_dim": int(feature_dim),
        "geodesic_flow_n_components": int(n_components),
        "geodesic_flow_n_bases": int(n_bases),
        "geodesic_flow_output_dim": int(n_components * n_bases),
        "geodesic_flow_requested_components": str(cfg.n_components),
        "geodesic_flow_n_steps": int(cfg.n_steps),
        "geodesic_flow_center": bool(cfg.center),
        "geodesic_flow_scale": bool(cfg.scale),
        "geodesic_flow_include_endpoints": bool(cfg.include_endpoints),
        "geodesic_flow_normalize_blocks": bool(cfg.normalize_blocks),
        "geodesic_flow_epsilon": float(cfg.epsilon),
    }


def _max_components(source: np.ndarray, target: np.ndarray, *, center: bool) -> int:
    return min(_max_single_domain_components(source, center=center), _max_single_domain_components(target, center=center))


def _max_single_domain_components(matrix: np.ndarray, *, center: bool) -> int:
    sample_limit = matrix.shape[0] - 1 if center and matrix.shape[0] > 1 else matrix.shape[0]
    return max(1, min(int(sample_limit), int(matrix.shape[1])))


def _effective_components(value: int | str | float, *, max_components: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("n_components must be a positive integer, 'all', or infinity.")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            requested = DEFAULT_GEODESIC_COMPONENTS
        elif text in {"all", "full", "inf", "infinity"}:
            requested = float("inf")
        else:
            requested = float(text)
    else:
        requested = float(value)
    if requested == float("inf"):
        return int(max_components)
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1.0:
        raise ValueError("n_components must be a positive integer, 'all', or infinity.")
    return min(int(requested), int(max_components))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _component_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = _feature_matrix(values, name=name)
    if matrix.shape[0] > matrix.shape[1]:
        raise ValueError(f"{name} should have shape n_components x n_features with n_components <= n_features.")
    return matrix


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


__all__ = [
    "GEODESIC_FLOW_CATEGORY",
    "GEODESIC_FLOW_PROTOCOL",
    "GeodesicFlowBasis",
    "GeodesicFlowConfig",
    "GeodesicFlowResult",
    "fit_sampled_geodesic_flow_features",
    "geodesic_flow_config",
    "sample_geodesic_bases",
    "transform_with_geodesic_bases",
]
