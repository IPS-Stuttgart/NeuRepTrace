"""Source-only robust feature scaling for cross-subject decoding.

This module fits feature-wise location and scale statistics from source rows only
and applies the fitted transform to source and evaluation feature matrices.  The
implementation is a strict Protocol-1 preprocessing helper: evaluation rows are
transformed but never used to fit the scaler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_ROBUST_SCALER_PROTOCOL = "strict_source_only_robust_feature_scaler"
SOURCE_ROBUST_SCALER_CATEGORY = "1_strict_source_only"
CENTER_MODES = ("median", "mean", "none")
SCALE_MODES = ("iqr", "mad", "std", "none")
DEFAULT_LOWER_QUANTILE = 0.25
DEFAULT_UPPER_QUANTILE = 0.75
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceRobustScalerConfig:
    """Configuration for source-only robust feature scaling."""

    center: str = "median"
    scale: str = "iqr"
    lower_quantile: float = DEFAULT_LOWER_QUANTILE
    upper_quantile: float = DEFAULT_UPPER_QUANTILE
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        """Validate direct dataclass construction as strictly as the public helper."""

        lower = _unit_interval_float(self.lower_quantile, name="lower_quantile")
        upper = _unit_interval_float(self.upper_quantile, name="upper_quantile")
        if lower >= upper:
            raise ValueError("lower_quantile must be smaller than upper_quantile.")
        object.__setattr__(self, "center", normalize_center_mode(self.center))
        object.__setattr__(self, "scale", normalize_scale_mode(self.scale))
        object.__setattr__(self, "lower_quantile", lower)
        object.__setattr__(self, "upper_quantile", upper)
        object.__setattr__(self, "epsilon", _positive_float(self.epsilon, name="epsilon"))


@dataclass(frozen=True, slots=True)
class SourceRobustScalerStats:
    """Fitted source-only scaler statistics."""

    location: np.ndarray
    scale: np.ndarray
    n_rows: int


@dataclass(frozen=True, slots=True)
class SourceRobustScalerResult:
    """Scaled feature matrices and fitted source-only statistics."""

    train_features: np.ndarray
    test_features: np.ndarray
    stats: SourceRobustScalerStats
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_robust_scaler(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceRobustScalerConfig | Mapping[str, Any] | None = None,
) -> SourceRobustScalerResult:
    """Fit robust scaler statistics on source rows and transform matrices."""

    cfg = source_robust_scaler_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    stats = fit_source_robust_scaler_stats(source, config=cfg)
    train = apply_source_robust_scaler(source, stats=stats)
    test_scaled = apply_source_robust_scaler(test, stats=stats)
    metadata = {
        "source_robust_scaler": True,
        "source_robust_scaler_protocol": SOURCE_ROBUST_SCALER_PROTOCOL,
        "source_robust_scaler_protocol_category": SOURCE_ROBUST_SCALER_CATEGORY,
        "source_robust_scaler_uses_source_features": True,
        "source_robust_scaler_uses_source_labels": False,
        "source_robust_scaler_uses_test_features_for_fitting": False,
        "source_robust_scaler_uses_test_labels": False,
        "source_robust_scaler_valid_for_strict_source_only": True,
        "source_robust_scaler_valid_for_benchmark": True,
        "source_robust_scaler_n_source_rows": int(source.shape[0]),
        "source_robust_scaler_n_test_rows": int(test.shape[0]),
        "source_robust_scaler_feature_dim": int(source.shape[1]),
        "source_robust_scaler_center": cfg.center,
        "source_robust_scaler_scale": cfg.scale,
        "source_robust_scaler_lower_quantile": float(cfg.lower_quantile),
        "source_robust_scaler_upper_quantile": float(cfg.upper_quantile),
        "source_robust_scaler_epsilon": float(cfg.epsilon),
    }
    return SourceRobustScalerResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_scaled.astype(np.float32, copy=False),
        stats=stats,
        metadata=metadata,
    )


def source_robust_scaler_config(
    *,
    center: str | None = "median",
    scale: str | None = "iqr",
    lower_quantile: float | str = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float | str = DEFAULT_UPPER_QUANTILE,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceRobustScalerConfig:
    """Normalize public robust-scaler options."""

    lower = _unit_interval_float(lower_quantile, name="lower_quantile")
    upper = _unit_interval_float(upper_quantile, name="upper_quantile")
    if lower >= upper:
        raise ValueError("lower_quantile must be smaller than upper_quantile.")
    return SourceRobustScalerConfig(
        center=normalize_center_mode(center),
        scale=normalize_scale_mode(scale),
        lower_quantile=lower,
        upper_quantile=upper,
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def fit_source_robust_scaler_stats(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceRobustScalerConfig | Mapping[str, Any] | None = None,
) -> SourceRobustScalerStats:
    """Fit source-only location and scale vectors."""

    cfg = source_robust_scaler_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    if cfg.center == "median":
        location = np.median(source, axis=0)
    elif cfg.center == "mean":
        location = np.mean(source, axis=0)
    elif cfg.center == "none":
        location = np.zeros(source.shape[1], dtype=float)
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unhandled center mode {cfg.center!r}.")

    if cfg.scale == "iqr":
        upper = np.quantile(source, cfg.upper_quantile, axis=0)
        lower = np.quantile(source, cfg.lower_quantile, axis=0)
        scale = upper - lower
    elif cfg.scale == "mad":
        med = np.median(source, axis=0)
        scale = 1.4826 * np.median(np.abs(source - med), axis=0)
    elif cfg.scale == "std":
        scale = np.std(source - np.mean(source, axis=0, keepdims=True), axis=0, ddof=1 if source.shape[0] > 1 else 0)
    elif cfg.scale == "none":
        scale = np.ones(source.shape[1], dtype=float)
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unhandled scale mode {cfg.scale!r}.")
    scale = np.maximum(np.asarray(scale, dtype=float), cfg.epsilon)
    return SourceRobustScalerStats(location=np.asarray(location, dtype=float), scale=scale, n_rows=int(source.shape[0]))


def apply_source_robust_scaler(features: Sequence[Sequence[float]] | np.ndarray, *, stats: SourceRobustScalerStats) -> np.ndarray:
    """Apply fitted source-only robust scaler statistics."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != stats.location.shape[0] or matrix.shape[1] != stats.scale.shape[0]:
        raise ValueError("features width must match fitted scaler statistics.")
    return (matrix - stats.location) / stats.scale


def normalize_center_mode(value: str | None) -> str:
    """Normalize center-mode aliases."""

    normalized = "median" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"med": "median", "average": "mean", "avg": "mean", "off": "none", "false": "none", "no": "none"}.get(normalized, normalized)
    if normalized not in CENTER_MODES:
        raise ValueError(f"Unknown center mode {value!r}. Available modes: {', '.join(CENTER_MODES)}.")
    return normalized


def normalize_scale_mode(value: str | None) -> str:
    """Normalize scale-mode aliases."""

    normalized = "iqr" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"interquartile": "iqr", "inter_quartile": "iqr", "median_absolute_deviation": "mad", "sd": "std", "off": "none", "false": "none", "no": "none"}.get(normalized, normalized)
    if normalized not in SCALE_MODES:
        raise ValueError(f"Unknown scale mode {value!r}. Available modes: {', '.join(SCALE_MODES)}.")
    return normalized


def _coerce_config(config: SourceRobustScalerConfig | Mapping[str, Any]) -> SourceRobustScalerConfig:
    if isinstance(config, SourceRobustScalerConfig):
        return config
    return source_robust_scaler_config(**dict(config))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _scalar_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a scalar finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar finite number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a scalar finite number.")
    return parsed


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _scalar_float(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _scalar_float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
