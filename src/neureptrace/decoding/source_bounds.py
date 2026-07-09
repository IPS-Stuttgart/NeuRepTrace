"""Strict source-only feature bounding utilities.

This module estimates feature-wise lower and upper bounds from source rows only
and applies those bounds to source rows and held-out rows.  It is intended as a
fold-local robust preprocessing helper for cross-subject decoding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_BOUNDS_PROTOCOL = "strict_source_only_feature_bounds"
SOURCE_BOUNDS_CATEGORY = "1_strict_source_only"
CENTER_MODES = ("median", "mean", "zero")
DEFAULT_LOWER_QUANTILE = 0.01
DEFAULT_UPPER_QUANTILE = 0.99


@dataclass(frozen=True, slots=True)
class SourceBoundsConfig:
    """Configuration for source-only feature bounds."""

    lower_quantile: float | str = DEFAULT_LOWER_QUANTILE
    upper_quantile: float | str = DEFAULT_UPPER_QUANTILE
    symmetric: bool | int | str = False
    center: str | None = "median"

    def __post_init__(self) -> None:
        lower = _unit_interval_float(self.lower_quantile, name="lower_quantile")
        upper = _unit_interval_float(self.upper_quantile, name="upper_quantile")
        if lower >= upper:
            raise ValueError("lower_quantile must be smaller than upper_quantile.")
        object.__setattr__(self, "lower_quantile", lower)
        object.__setattr__(self, "upper_quantile", upper)
        object.__setattr__(self, "symmetric", _bool_value(self.symmetric, name="symmetric"))
        object.__setattr__(self, "center", normalize_bounds_center(self.center))


@dataclass(frozen=True, slots=True)
class SourceFeatureBounds:
    """Feature-wise bounds fitted from source rows."""

    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceBoundsResult:
    """Bounded train/test rows and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    bounds: SourceFeatureBounds
    train_changed_mask: np.ndarray
    test_changed_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_feature_bounds(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceBoundsConfig | Mapping[str, Any] | None = None,
) -> SourceBoundsResult:
    """Fit source-only feature bounds and transform source/test rows.

    The held-out rows are transformed with source-fitted bounds only.  They are
    not used to estimate the bounds.
    """

    cfg = source_bounds_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    bounds = fit_source_feature_bound_values(source, config=cfg)
    train_features, train_mask = apply_source_feature_bounds(source, bounds)
    test_features_out, test_mask = apply_source_feature_bounds(test, bounds)
    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_test_rows=test.shape[0],
        feature_dim=source.shape[1],
        train_mask=train_mask,
        test_mask=test_mask,
    )
    return SourceBoundsResult(
        train_features=train_features.astype(np.float32, copy=False),
        test_features=test_features_out.astype(np.float32, copy=False),
        bounds=bounds,
        train_changed_mask=train_mask,
        test_changed_mask=test_mask,
        metadata=metadata,
    )


def source_bounds_config(
    *,
    lower_quantile: float | str = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float | str = DEFAULT_UPPER_QUANTILE,
    symmetric: bool | int | str = False,
    center: str | None = "median",
) -> SourceBoundsConfig:
    """Normalize public feature-bound options."""

    lower = _unit_interval_float(lower_quantile, name="lower_quantile")
    upper = _unit_interval_float(upper_quantile, name="upper_quantile")
    if lower >= upper:
        raise ValueError("lower_quantile must be smaller than upper_quantile.")
    return SourceBoundsConfig(
        lower_quantile=lower,
        upper_quantile=upper,
        symmetric=_bool_value(symmetric, name="symmetric"),
        center=normalize_bounds_center(center),
    )


def normalize_bounds_center(value: str | None) -> str:
    """Normalize center aliases used by symmetric bounds."""

    normalized = "median" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"med": "median", "average": "mean", "zero_center": "zero", "none": "zero"}.get(normalized, normalized)
    if normalized not in CENTER_MODES:
        raise ValueError(f"Unknown center mode {value!r}. Available modes: {', '.join(CENTER_MODES)}.")
    return normalized


def fit_source_feature_bound_values(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceBoundsConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureBounds:
    """Estimate feature-wise bounds from source rows only."""

    cfg = source_bounds_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    center = _center_vector(source, mode=cfg.center)
    if cfg.symmetric:
        deviations = np.abs(source - center)
        radius = np.quantile(deviations, cfg.upper_quantile, axis=0)
        lower = center - radius
        upper = center + radius
    else:
        lower = np.quantile(source, cfg.lower_quantile, axis=0)
        upper = np.quantile(source, cfg.upper_quantile, axis=0)
    lower, upper = _repair_bounds(lower, upper)
    return SourceFeatureBounds(
        lower=lower.astype(float, copy=False),
        upper=upper.astype(float, copy=False),
        center=center.astype(float, copy=False),
        n_source_rows=int(source.shape[0]),
    )


def apply_source_feature_bounds(features: Sequence[Sequence[float]] | np.ndarray, bounds: SourceFeatureBounds) -> tuple[np.ndarray, np.ndarray]:
    """Apply source-fitted bounds and return a changed-value mask."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != bounds.lower.shape[0] or matrix.shape[1] != bounds.upper.shape[0]:
        raise ValueError("features width must match bound vectors.")
    bounded = np.minimum(np.maximum(matrix, bounds.lower), bounds.upper)
    return bounded, bounded != matrix


def _coerce_config(config: SourceBoundsConfig | Mapping[str, Any]) -> SourceBoundsConfig:
    if isinstance(config, SourceBoundsConfig):
        return source_bounds_config(
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            symmetric=config.symmetric,
            center=config.center,
        )
    return source_bounds_config(**dict(config))


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
    cfg: SourceBoundsConfig,
    *,
    n_source_rows: int,
    n_test_rows: int,
    feature_dim: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "source_feature_bounds": True,
        "source_feature_bounds_protocol": SOURCE_BOUNDS_PROTOCOL,
        "source_feature_bounds_protocol_category": SOURCE_BOUNDS_CATEGORY,
        "source_feature_bounds_uses_source_features": True,
        "source_feature_bounds_uses_test_features_for_fitting": False,
        "source_feature_bounds_uses_test_labels": False,
        "source_feature_bounds_valid_for_strict_source_only": True,
        "source_feature_bounds_valid_for_benchmark": True,
        "source_feature_bounds_n_source_rows": int(n_source_rows),
        "source_feature_bounds_n_test_rows": int(n_test_rows),
        "source_feature_bounds_feature_dim": int(feature_dim),
        "source_feature_bounds_lower_quantile": float(cfg.lower_quantile),
        "source_feature_bounds_upper_quantile": float(cfg.upper_quantile),
        "source_feature_bounds_symmetric": bool(cfg.symmetric),
        "source_feature_bounds_center": cfg.center,
        "source_feature_bounds_train_values_changed": int(np.count_nonzero(train_mask)),
        "source_feature_bounds_test_values_changed": int(np.count_nonzero(test_mask)),
    }


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize nested one-pass iterables before NumPy consumes them."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        materialized = [_materialize_one_pass_iterables(item) for item in value.ravel(order="C")]
        return np.asarray(materialized, dtype=object).reshape(value.shape)
    if isinstance(value, (str, bytes)):
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
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Iterable):
        return False
    return any(_contains_boolean_value(item) for item in value)


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


def _scalar_array_value(value: object, *, name: str) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar value.")
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unit_interval_float(value: float | str, *, name: str) -> float:
    value = _scalar_array_value(value, name=name)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric quantile, not boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1].") from exc
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _bool_value(value: bool | int | str, *, name: str) -> bool:
    value = _scalar_array_value(value, name=name)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")
