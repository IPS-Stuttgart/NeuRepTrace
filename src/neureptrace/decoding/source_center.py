"""Strict source-only feature centering.

This module estimates a feature-wise center from source rows only and applies the
fixed center to source and held-out rows.  It is a small Protocol-1 preprocessing
helper for fold-local cross-subject decoding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_CENTER_PROTOCOL = "strict_source_only_feature_centering"
SOURCE_CENTER_CATEGORY = "1_strict_source_only"
CENTER_MODES = ("mean", "median", "zero")


@dataclass(frozen=True, slots=True)
class SourceCenterConfig:
    """Configuration for source-only feature centering."""

    center: str = "mean"


@dataclass(frozen=True, slots=True)
class SourceCenterMap:
    """Feature-wise source center."""

    center: np.ndarray
    center_mode: str
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceCenterResult:
    """Centered source/test features and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    center_map: SourceCenterMap
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_center_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceCenterConfig | Mapping[str, Any] | None = None,
) -> SourceCenterResult:
    """Fit a source-only center and transform source/test rows."""

    cfg = source_center_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    center_map = fit_source_center_map(source, config=cfg)
    train = apply_source_center_transform(source, center_map)
    test_out = apply_source_center_transform(test, center_map)
    return SourceCenterResult(
        train_features=_float32_if_safe(train),
        test_features=_float32_if_safe(test_out),
        center_map=center_map,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1]),
    )


def source_center_config(*, center: str | None = "mean") -> SourceCenterConfig:
    """Normalize source-centering options."""

    return SourceCenterConfig(center=normalize_center_mode(center))


def normalize_center_mode(value: str | None) -> str:
    """Normalize center-mode aliases."""

    normalized = "mean" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"avg": "mean", "average": "mean", "med": "median", "none": "zero", "constant_zero": "zero"}.get(normalized, normalized)
    if normalized not in CENTER_MODES:
        raise ValueError(f"Unknown center mode {value!r}. Available values: {', '.join(CENTER_MODES)}.")
    return normalized


def fit_source_center_map(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceCenterConfig | Mapping[str, Any] | None = None,
) -> SourceCenterMap:
    """Estimate a feature-wise source center."""

    cfg = source_center_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    if cfg.center == "mean":
        center = _stable_feature_mean(source)
    elif cfg.center == "median":
        center = np.median(source, axis=0)
    elif cfg.center == "zero":
        center = np.zeros(source.shape[1], dtype=float)
    else:  # pragma: no cover - normalized above
        raise ValueError(f"Unhandled center mode {cfg.center!r}.")
    return SourceCenterMap(center=center.astype(float, copy=False), center_mode=cfg.center, n_source_rows=int(source.shape[0]))


def apply_source_center_transform(features: Sequence[Sequence[float]] | np.ndarray, center_map: SourceCenterMap) -> np.ndarray:
    """Subtract a source-fitted center from feature rows."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != center_map.center.shape[0]:
        raise ValueError("features width must match center map width.")
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = matrix - center_map.center[None, :]
    if not np.all(np.isfinite(transformed)):
        raise ValueError("source-centering output must contain only finite values.")
    return transformed


def _stable_feature_mean(source: np.ndarray) -> np.ndarray:
    """Compute feature means without overflowing on finite high-magnitude rows."""

    scale = np.max(np.abs(source), axis=0)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    return np.mean(source / safe_scale[None, :], axis=0) * scale


def _float32_if_safe(values: Any) -> np.ndarray:
    """Use float32 unless conversion overflows or erases nonzero values."""

    array = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    lost_finite = np.isfinite(array) & ~np.isfinite(compact)
    lost_nonzero = (array != 0.0) & (compact == 0.0)
    if bool(np.any(lost_finite | lost_nonzero)):
        return array
    return compact


def _coerce_config(config: SourceCenterConfig | Mapping[str, Any]) -> SourceCenterConfig:
    if isinstance(config, SourceCenterConfig):
        return config
    return source_center_config(**dict(config))


def _metadata(cfg: SourceCenterConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_center_transform": True,
        "source_center_protocol": SOURCE_CENTER_PROTOCOL,
        "source_center_protocol_category": SOURCE_CENTER_CATEGORY,
        "source_center_uses_source_features": True,
        "source_center_uses_test_features_for_fitting": False,
        "source_center_uses_labels": False,
        "source_center_valid_for_strict_source_only": True,
        "source_center_valid_for_benchmark": True,
        "source_center_n_source_rows": int(n_source_rows),
        "source_center_n_test_rows": int(n_test_rows),
        "source_center_feature_dim": int(feature_dim),
        "source_center_mode": cfg.center,
    }


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize generator-backed feature rows before NumPy conversion."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        materialized = [_materialize_one_pass_iterables(item) for item in value.ravel(order="C")]
        return np.asarray(materialized, dtype=object).reshape(value.shape)
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__array__"):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_one_pass_iterables(item) for item in value]


def _contains_boolean_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if hasattr(value, "__array__"):
        try:
            return _contains_boolean_value(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, Iterable):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")
    try:
        matrix = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix
