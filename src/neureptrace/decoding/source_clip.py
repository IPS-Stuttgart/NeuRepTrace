"""Strict source-only feature clipping.

This module estimates feature-wise clipping bounds from source rows only and then
applies those bounds to train/test matrices.  It is a lightweight preprocessing
baseline for cross-subject decoding when extreme feature values can destabilize
fold-local classifiers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_CLIP_PROTOCOL = "strict_source_only_feature_clipping"
SOURCE_CLIP_CATEGORY = "1_strict_source_only"
CENTER_MODES = ("median", "mean", "zero")
DEFAULT_LOWER_QUANTILE = 0.01
DEFAULT_UPPER_QUANTILE = 0.99


@dataclass(frozen=True, slots=True)
class SourceClipConfig:
    """Configuration for source-fitted clipping."""

    lower_quantile: float = DEFAULT_LOWER_QUANTILE
    upper_quantile: float = DEFAULT_UPPER_QUANTILE
    symmetric: bool = False
    center: str = "median"


@dataclass(frozen=True, slots=True)
class SourceClipBounds:
    """Feature-wise clipping bounds fitted from source rows."""

    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceClipResult:
    """Clipped train/test features and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    bounds: SourceClipBounds
    train_clipped_mask: np.ndarray
    test_clipped_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_clip(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceClipConfig | Mapping[str, Any] | None = None,
) -> SourceClipResult:
    """Fit clipping bounds on source rows and transform source/test rows."""

    cfg = source_clip_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    bounds = fit_source_clip_bounds(source, config=cfg)
    train_clipped, train_mask = apply_source_clip(source, bounds)
    test_clipped, test_mask = apply_source_clip(test, bounds)
    return SourceClipResult(
        train_features=train_clipped.astype(np.float32, copy=False),
        test_features=test_clipped.astype(np.float32, copy=False),
        bounds=bounds,
        train_clipped_mask=train_mask,
        test_clipped_mask=test_mask,
        metadata=_metadata(
            cfg,
            n_source_rows=source.shape[0],
            n_test_rows=test.shape[0],
            feature_dim=source.shape[1],
            train_mask=train_mask,
            test_mask=test_mask,
        ),
    )


def source_clip_config(
    *,
    lower_quantile: float | str = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float | str = DEFAULT_UPPER_QUANTILE,
    symmetric: bool | str | int | float = False,
    center: str | None = "median",
) -> SourceClipConfig:
    """Normalize public clipping options."""

    lower = _unit_interval_float(lower_quantile, name="lower_quantile")
    upper = _unit_interval_float(upper_quantile, name="upper_quantile")
    if lower >= upper:
        raise ValueError("lower_quantile must be smaller than upper_quantile.")
    return SourceClipConfig(
        lower_quantile=lower,
        upper_quantile=upper,
        symmetric=_bool_config(symmetric, name="symmetric"),
        center=normalize_center_mode(center),
    )


def normalize_center_mode(value: str | None) -> str:
    """Normalize center aliases for symmetric clipping."""

    normalized = "median" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"med": "median", "avg": "mean", "average": "mean", "none": "zero"}.get(normalized, normalized)
    if normalized not in CENTER_MODES:
        raise ValueError(f"Unknown center mode {value!r}. Available values: {', '.join(CENTER_MODES)}.")
    return normalized


def fit_source_clip_bounds(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceClipConfig | Mapping[str, Any] | None = None,
) -> SourceClipBounds:
    """Estimate feature-wise clipping bounds from source rows only."""

    cfg = source_clip_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    center = _center_vector(source, mode=cfg.center)
    if cfg.symmetric:
        radius = np.quantile(np.abs(source - center), cfg.upper_quantile, axis=0)
        lower = center - radius
        upper = center + radius
    else:
        lower = np.quantile(source, cfg.lower_quantile, axis=0)
        upper = np.quantile(source, cfg.upper_quantile, axis=0)
    lower, upper = _repair_bounds(lower, upper)
    return SourceClipBounds(
        lower=lower.astype(float, copy=False),
        upper=upper.astype(float, copy=False),
        center=center.astype(float, copy=False),
        n_source_rows=int(source.shape[0]),
    )


def apply_source_clip(features: Sequence[Sequence[float]] | np.ndarray, bounds: SourceClipBounds) -> tuple[np.ndarray, np.ndarray]:
    """Apply source-fitted clipping bounds and return a changed-value mask."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != bounds.lower.shape[0] or matrix.shape[1] != bounds.upper.shape[0]:
        raise ValueError("features width must match clipping bounds.")
    clipped = np.clip(matrix, bounds.lower, bounds.upper)
    return clipped, clipped != matrix


def _coerce_config(config: SourceClipConfig | Mapping[str, Any]) -> SourceClipConfig:
    if isinstance(config, SourceClipConfig):
        return config
    return source_clip_config(**dict(config))


def _center_vector(source: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "median":
        return np.median(source, axis=0)
    if mode == "mean":
        return np.mean(source, axis=0)
    if mode == "zero":
        return np.zeros(source.shape[1], dtype=float)
    raise ValueError(f"Unhandled center mode {mode!r}.")


def _repair_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fixed_lower = np.minimum(lower, upper)
    fixed_upper = np.maximum(lower, upper)
    equal = fixed_upper <= fixed_lower
    if np.any(equal):
        fixed_upper = fixed_upper.copy()
        fixed_upper[equal] = fixed_lower[equal] + np.finfo(float).eps
    return fixed_lower, fixed_upper


def _metadata(
    cfg: SourceClipConfig,
    *,
    n_source_rows: int,
    n_test_rows: int,
    feature_dim: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "source_clip": True,
        "source_clip_protocol": SOURCE_CLIP_PROTOCOL,
        "source_clip_protocol_category": SOURCE_CLIP_CATEGORY,
        "source_clip_uses_source_features": True,
        "source_clip_uses_test_features_for_fitting": False,
        "source_clip_uses_test_labels": False,
        "source_clip_valid_for_strict_source_only": True,
        "source_clip_valid_for_benchmark": True,
        "source_clip_n_source_rows": int(n_source_rows),
        "source_clip_n_test_rows": int(n_test_rows),
        "source_clip_feature_dim": int(feature_dim),
        "source_clip_lower_quantile": float(cfg.lower_quantile),
        "source_clip_upper_quantile": float(cfg.upper_quantile),
        "source_clip_symmetric": bool(cfg.symmetric),
        "source_clip_center": cfg.center,
        "source_clip_train_values_clipped": int(np.count_nonzero(train_mask)),
        "source_clip_test_values_clipped": int(np.count_nonzero(test_mask)),
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


def _unit_interval_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1].") from exc
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed
