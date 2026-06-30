"""Strict source-only feature scaling helpers.

This module fits simple feature-wise scaling statistics on source rows only and
applies the fitted transform to source and test matrices.  It is intended as a
small Protocol-1 preprocessing helper for fold-local decoding pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_SCALE_PROTOCOL = "strict_source_only_feature_scaling"
SOURCE_SCALE_CATEGORY = "1_strict_source_only"
SCALE_METHODS = ("standard", "robust", "minmax")
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceFeatureScaleConfig:
    """Configuration for source-only feature scaling."""

    method: str = "standard"
    center: bool = True
    scale: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceFeatureScaleStats:
    """Feature-wise source statistics used for scaling."""

    offset: np.ndarray
    scale: np.ndarray
    method: str
    n_fit_rows: int


@dataclass(frozen=True, slots=True)
class SourceFeatureScaleResult:
    """Scaled source/test matrices and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    stats: SourceFeatureScaleStats
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_feature_scale(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceFeatureScaleConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureScaleResult:
    """Fit source-only scaling statistics and transform source/test rows."""

    cfg = source_feature_scale_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    stats = fit_source_feature_scale_stats(source, config=cfg)
    train = apply_source_feature_scale(source, stats)
    test_out = apply_source_feature_scale(test, stats)
    return SourceFeatureScaleResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        stats=stats,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1]),
    )


def fit_source_feature_scale_stats(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceFeatureScaleConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureScaleStats:
    """Estimate feature-wise scaling statistics from source rows only."""

    cfg = source_feature_scale_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    if cfg.method == "standard":
        offset = np.mean(source, axis=0) if cfg.center else np.zeros(source.shape[1], dtype=float)
        scale = np.std(source - offset, axis=0, ddof=1 if source.shape[0] > 1 else 0) if cfg.scale else np.ones(source.shape[1], dtype=float)
    elif cfg.method == "robust":
        offset = np.median(source, axis=0) if cfg.center else np.zeros(source.shape[1], dtype=float)
        q75 = np.percentile(source, 75.0, axis=0)
        q25 = np.percentile(source, 25.0, axis=0)
        scale = (q75 - q25) / 1.349 if cfg.scale else np.ones(source.shape[1], dtype=float)
    elif cfg.method == "minmax":
        minimum = np.min(source, axis=0)
        maximum = np.max(source, axis=0)
        offset = minimum if cfg.center else np.zeros(source.shape[1], dtype=float)
        scale = (maximum - minimum) if cfg.scale else np.ones(source.shape[1], dtype=float)
    else:  # pragma: no cover - guarded by config normalization
        raise ValueError(f"Unhandled source scaling method {cfg.method!r}.")
    scale = np.maximum(np.asarray(scale, dtype=float), cfg.epsilon)
    return SourceFeatureScaleStats(offset=np.asarray(offset, dtype=float), scale=scale, method=cfg.method, n_fit_rows=int(source.shape[0]))


def apply_source_feature_scale(features: Sequence[Sequence[float]] | np.ndarray, stats: SourceFeatureScaleStats) -> np.ndarray:
    """Apply fitted source-only scaling statistics."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != stats.offset.shape[0] or matrix.shape[1] != stats.scale.shape[0]:
        raise ValueError("features width must match fitted source scaling statistics.")
    return (matrix - stats.offset) / stats.scale


def source_feature_scale_config(
    *,
    method: str | None = "standard",
    center: bool | str | int = True,
    scale: bool | str | int = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceFeatureScaleConfig:
    """Normalize public source-scaling options."""

    return SourceFeatureScaleConfig(
        method=normalize_source_scale_method(method),
        center=_bool_value(center, name="center"),
        scale=_bool_value(scale, name="scale"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_source_scale_method(value: str | None) -> str:
    """Normalize scaling method aliases."""

    normalized = "standard" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"zscore": "standard", "z_score": "standard", "iqr": "robust", "median_iqr": "robust", "range": "minmax", "min_max": "minmax"}.get(normalized, normalized)
    if normalized not in SCALE_METHODS:
        raise ValueError(f"Unknown source scale method {value!r}. Available methods: {', '.join(SCALE_METHODS)}.")
    return normalized


def _coerce_config(config: SourceFeatureScaleConfig | Mapping[str, Any]) -> SourceFeatureScaleConfig:
    if isinstance(config, SourceFeatureScaleConfig):
        return config
    return source_feature_scale_config(**dict(config))


def _metadata(cfg: SourceFeatureScaleConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_feature_scale": True,
        "source_feature_scale_protocol": SOURCE_SCALE_PROTOCOL,
        "source_feature_scale_protocol_category": SOURCE_SCALE_CATEGORY,
        "source_feature_scale_method": cfg.method,
        "source_feature_scale_uses_source_features": True,
        "source_feature_scale_uses_test_features_for_fitting": False,
        "source_feature_scale_uses_test_labels": False,
        "source_feature_scale_valid_for_strict_source_only": True,
        "source_feature_scale_valid_for_benchmark": True,
        "source_feature_scale_n_source_rows": int(n_source_rows),
        "source_feature_scale_n_test_rows": int(n_test_rows),
        "source_feature_scale_feature_dim": int(feature_dim),
        "source_feature_scale_center": bool(cfg.center),
        "source_feature_scale_scale": bool(cfg.scale),
        "source_feature_scale_epsilon": float(cfg.epsilon),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_, np.ndarray)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _bool_value(value: bool | str | int, *, name: str) -> bool:
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
