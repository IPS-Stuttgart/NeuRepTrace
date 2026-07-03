"""Strict source-only feature standardization.

This module fits feature-wise location and scale statistics from source rows only
and applies them to source and held-out rows.  It is a fold-local preprocessing
helper for cross-subject decoding that keeps the held-out subject out of the fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_STANDARDIZE_PROTOCOL = "strict_source_only_feature_standardization"
SOURCE_STANDARDIZE_CATEGORY = "1_strict_source_only"
LOCATION_MODES = ("mean", "median", "zero")
SCALE_MODES = ("std", "iqr", "mad", "none")
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceStandardizeConfig:
    """Configuration for source-fitted feature standardization."""

    location: str = "mean"
    scale: str = "std"
    clip: float | None = None
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", normalize_location_mode(self.location))
        object.__setattr__(self, "scale", normalize_scale_mode(self.scale))
        object.__setattr__(self, "clip", _optional_positive_float(self.clip, name="clip"))
        object.__setattr__(self, "epsilon", _positive_float(self.epsilon, name="epsilon"))


@dataclass(frozen=True, slots=True)
class SourceStandardizeStats:
    """Source-fitted location and scale vectors."""

    location: np.ndarray
    scale: np.ndarray
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceStandardizeResult:
    """Standardized source/test features and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    stats: SourceStandardizeStats
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_standardizer(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceStandardizeConfig | Mapping[str, Any] | None = None,
) -> SourceStandardizeResult:
    """Fit source-only standardization stats and transform source/test rows."""

    cfg = source_standardize_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    stats = fit_source_standardize_stats(source, config=cfg)
    train = transform_with_source_standardizer(source, stats, config=cfg)
    test_out = transform_with_source_standardizer(test, stats, config=cfg)
    return SourceStandardizeResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        stats=stats,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1]),
    )


def source_standardize_config(
    *,
    location: str | None = "mean",
    scale: str | None = "std",
    clip: float | str | None = None,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceStandardizeConfig:
    """Normalize public standardization options."""

    return SourceStandardizeConfig(
        location=normalize_location_mode(location),
        scale=normalize_scale_mode(scale),
        clip=_optional_positive_float(clip, name="clip"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_location_mode(value: str | None) -> str:
    """Normalize location-mode aliases."""

    scalar = _scalar_value(value, name="location mode")
    normalized = "mean" if scalar is None else str(scalar).strip().lower().replace("-", "_")
    normalized = {"avg": "mean", "average": "mean", "med": "median", "none": "zero", "off": "zero"}.get(normalized, normalized)
    if normalized not in LOCATION_MODES:
        raise ValueError(f"Unknown location mode {value!r}. Available values: {', '.join(LOCATION_MODES)}.")
    return normalized


def normalize_scale_mode(value: str | None) -> str:
    """Normalize scale-mode aliases."""

    scalar = _scalar_value(value, name="scale mode")
    normalized = "std" if scalar is None else str(scalar).strip().lower().replace("-", "_")
    normalized = {"standard": "std", "sd": "std", "iqr_scale": "iqr", "median_abs_deviation": "mad", "off": "none", "unit": "none"}.get(normalized, normalized)
    if normalized not in SCALE_MODES:
        raise ValueError(f"Unknown scale mode {value!r}. Available values: {', '.join(SCALE_MODES)}.")
    return normalized


def fit_source_standardize_stats(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceStandardizeConfig | Mapping[str, Any] | None = None,
) -> SourceStandardizeStats:
    """Fit location and scale vectors from source rows only."""

    cfg = source_standardize_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    location = _location_vector(source, mode=cfg.location)
    scale = _scale_vector(source, location=location, mode=cfg.scale, epsilon=cfg.epsilon)
    return SourceStandardizeStats(location=location.astype(float, copy=False), scale=scale.astype(float, copy=False), n_source_rows=int(source.shape[0]))


def transform_with_source_standardizer(
    features: Sequence[Sequence[float]] | np.ndarray,
    stats: SourceStandardizeStats,
    *,
    config: SourceStandardizeConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply source-fitted standardization stats."""

    cfg = source_standardize_config() if config is None else _coerce_config(config)
    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != stats.location.shape[0] or matrix.shape[1] != stats.scale.shape[0]:
        raise ValueError("features width must match source standardization statistics.")
    transformed = (matrix - stats.location) / stats.scale
    if cfg.clip is not None:
        transformed = np.clip(transformed, -cfg.clip, cfg.clip)
    return transformed


def _coerce_config(config: SourceStandardizeConfig | Mapping[str, Any]) -> SourceStandardizeConfig:
    if isinstance(config, SourceStandardizeConfig):
        return config
    return source_standardize_config(**dict(config))


def _location_vector(source: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.mean(source, axis=0)
    if mode == "median":
        return np.median(source, axis=0)
    if mode == "zero":
        return np.zeros(source.shape[1], dtype=float)
    raise ValueError(f"Unhandled location mode {mode!r}.")


def _scale_vector(source: np.ndarray, *, location: np.ndarray, mode: str, epsilon: float) -> np.ndarray:
    if mode == "none":
        return np.ones(source.shape[1], dtype=float)
    if mode == "std":
        scale = np.std(source - location, axis=0, ddof=1 if source.shape[0] > 1 else 0)
    elif mode == "iqr":
        scale = (np.percentile(source, 75.0, axis=0) - np.percentile(source, 25.0, axis=0)) / 1.349
    elif mode == "mad":
        scale = 1.4826 * np.median(np.abs(source - location), axis=0)
    else:
        raise ValueError(f"Unhandled scale mode {mode!r}.")
    return np.maximum(np.asarray(scale, dtype=float), float(epsilon))


def _metadata(cfg: SourceStandardizeConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_standardizer": True,
        "source_standardizer_protocol": SOURCE_STANDARDIZE_PROTOCOL,
        "source_standardizer_protocol_category": SOURCE_STANDARDIZE_CATEGORY,
        "source_standardizer_uses_source_features": True,
        "source_standardizer_uses_test_features_for_fitting": False,
        "source_standardizer_uses_test_labels": False,
        "source_standardizer_valid_for_strict_source_only": True,
        "source_standardizer_valid_for_benchmark": True,
        "source_standardizer_n_source_rows": int(n_source_rows),
        "source_standardizer_n_test_rows": int(n_test_rows),
        "source_standardizer_feature_dim": int(feature_dim),
        "source_standardizer_location": cfg.location,
        "source_standardizer_scale": cfg.scale,
        "source_standardizer_clip": "" if cfg.clip is None else float(cfg.clip),
        "source_standardizer_epsilon": float(cfg.epsilon),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _scalar_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar.")
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _positive_float(value: float | str, *, name: str) -> float:
    scalar = _scalar_value(value, name=name)
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _optional_positive_float(value: float | str | None, *, name: str) -> float | None:
    scalar = _scalar_value(value, name=name)
    if scalar is None:
        return None
    if isinstance(scalar, str) and scalar.strip().lower() in {"", "none", "null"}:
        return None
    return _positive_float(scalar, name=name)
