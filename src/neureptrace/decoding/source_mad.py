"""Strict source-only median/MAD normalization helpers.

The helpers in this module estimate robust feature-wise center and scale from
source rows only, then apply the frozen transform to source and held-out rows.
This is a Protocol-1 preprocessing helper: held-out rows are transformed but not
used to estimate statistics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_MAD_PROTOCOL = "strict_source_only_mad_normalization"
SOURCE_MAD_CATEGORY = "1_strict_source_only"
DEFAULT_EPSILON = 1e-8
MAD_NORMAL_CONSTANT = 1.4826


@dataclass(frozen=True, slots=True)
class SourceMADConfig:
    """Configuration for source-only MAD normalization."""

    center: bool = True
    scale: bool = True
    normal_consistency: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceMADReference:
    """Source-fitted robust feature statistics."""

    center: np.ndarray
    scale: np.ndarray
    config: SourceMADConfig
    n_fit_rows: int


@dataclass(frozen=True, slots=True)
class SourceMADResult:
    """Transformed source/test features and metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourceMADReference
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_mad_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceMADConfig | Mapping[str, Any] | None = None,
) -> SourceMADResult:
    """Fit source MAD statistics and transform source/test rows."""

    cfg = source_mad_config() if config is None else _coerce_config(config)
    source = _matrix(source_features, name="source_features")
    test = _matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError("source_features and test_features must have the same feature width.")
    reference = fit_source_mad_reference(source, config=cfg)
    train = apply_source_mad_transform(source, reference)
    test_out = apply_source_mad_transform(test, reference)
    return SourceMADResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        reference=reference,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1]),
    )


def fit_source_mad_reference(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceMADConfig | Mapping[str, Any] | None = None,
) -> SourceMADReference:
    """Estimate robust feature statistics from source rows only."""

    cfg = source_mad_config() if config is None else _coerce_config(config)
    source = _matrix(source_features, name="source_features")
    source_median = np.median(source, axis=0)
    center = source_median if cfg.center else np.zeros(source.shape[1], dtype=float)
    raw_mad = np.median(np.abs(source - source_median), axis=0)
    scale = raw_mad * (MAD_NORMAL_CONSTANT if cfg.normal_consistency else 1.0) if cfg.scale else np.ones(source.shape[1], dtype=float)
    scale = np.maximum(scale, cfg.epsilon)
    return SourceMADReference(center=center.astype(float, copy=False), scale=scale.astype(float, copy=False), config=cfg, n_fit_rows=int(source.shape[0]))


def apply_source_mad_transform(features: Sequence[Sequence[float]] | np.ndarray, reference: SourceMADReference) -> np.ndarray:
    """Apply source-fitted MAD normalization."""

    matrix = _matrix(features, name="features")
    if matrix.shape[1] != reference.center.shape[0] or matrix.shape[1] != reference.scale.shape[0]:
        raise ValueError("features width must match source MAD reference.")
    return (matrix - reference.center) / reference.scale


def source_mad_config(
    *,
    center: bool | int | str = True,
    scale: bool | int | str = True,
    normal_consistency: bool | int | str = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceMADConfig:
    """Normalize source-MAD options."""

    return SourceMADConfig(
        center=_bool_value(center, name="center"),
        scale=_bool_value(scale, name="scale"),
        normal_consistency=_bool_value(normal_consistency, name="normal_consistency"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _coerce_config(config: SourceMADConfig | Mapping[str, Any]) -> SourceMADConfig:
    if isinstance(config, SourceMADConfig):
        return source_mad_config(
            center=config.center,
            scale=config.scale,
            normal_consistency=config.normal_consistency,
            epsilon=config.epsilon,
        )
    return source_mad_config(**dict(config))


def _metadata(cfg: SourceMADConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_mad": True,
        "source_mad_protocol": SOURCE_MAD_PROTOCOL,
        "source_mad_protocol_category": SOURCE_MAD_CATEGORY,
        "source_mad_uses_source_features": True,
        "source_mad_uses_test_features_for_fitting": False,
        "source_mad_uses_test_labels": False,
        "source_mad_valid_for_strict_source_only": True,
        "source_mad_valid_for_benchmark": True,
        "source_mad_n_source_rows": int(n_source_rows),
        "source_mad_n_test_rows": int(n_test_rows),
        "source_mad_feature_dim": int(feature_dim),
        "source_mad_center": bool(cfg.center),
        "source_mad_scale": bool(cfg.scale),
        "source_mad_normal_consistency": bool(cfg.normal_consistency),
        "source_mad_epsilon": float(cfg.epsilon),
    }


def _matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
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


def _bool_value(value: bool | int | str, *, name: str) -> bool:
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
