"""Strict source-only feature clipping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_CLIP_PROTOCOL = "strict_source_only_feature_clipping"
SOURCE_CLIP_STANDARDIZE_PROTOCOL = "strict_source_only_feature_clip_standardize"
SOURCE_CLIP_CATEGORY = "1_strict_source_only"
CENTER_MODES = ("median", "mean", "zero")
DEFAULT_LOWER_QUANTILE = 0.01
DEFAULT_UPPER_QUANTILE = 0.99
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceClipConfig:
    lower_quantile: float = DEFAULT_LOWER_QUANTILE
    upper_quantile: float = DEFAULT_UPPER_QUANTILE
    symmetric: bool = False
    center: str = "median"

    def __post_init__(self) -> None:
        lower = _unit_interval_float(self.lower_quantile, name="lower_quantile")
        upper = _unit_interval_float(self.upper_quantile, name="upper_quantile")
        if lower >= upper:
            raise ValueError("lower_quantile must be smaller than upper_quantile.")
        object.__setattr__(self, "lower_quantile", lower)
        object.__setattr__(self, "upper_quantile", upper)
        object.__setattr__(self, "symmetric", _bool_config(self.symmetric, name="symmetric"))
        object.__setattr__(self, "center", normalize_center_mode(self.center))


@dataclass(frozen=True, slots=True)
class SourceClipBounds:
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceClipResult:
    train_features: np.ndarray
    test_features: np.ndarray
    bounds: SourceClipBounds
    train_clipped_mask: np.ndarray
    test_clipped_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceClipStandardizeResult:
    train_features: np.ndarray
    test_features: np.ndarray
    clip_result: SourceClipResult
    center: np.ndarray
    scale: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_clip(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceClipConfig | Mapping[str, Any] | None = None,
) -> SourceClipResult:
    cfg = source_clip_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    bounds = fit_source_clip_bounds(source, config=cfg)
    train_clipped, train_mask = apply_source_clip(source, bounds)
    test_clipped, test_mask = apply_source_clip(test, bounds)
    return SourceClipResult(
        train_features=_compact_float32(train_clipped),
        test_features=_compact_float32(test_clipped),
        bounds=bounds,
        train_clipped_mask=train_mask,
        test_clipped_mask=test_mask,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], train_mask=train_mask, test_mask=test_mask),
    )


def fit_source_clip_then_standardize(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceClipConfig | Mapping[str, Any] | None = None,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceClipStandardizeResult:
    """Clip with source-fitted bounds, then standardize with clipped source rows."""

    eps = _positive_float(epsilon, name="epsilon")
    clipped = fit_source_clip(source_features=source_features, test_features=test_features, config=config)
    center = np.mean(clipped.train_features, axis=0)
    scale = np.std(clipped.train_features - center, axis=0, ddof=1 if clipped.train_features.shape[0] > 1 else 0)
    scale = np.maximum(scale, eps)
    train_scaled = (clipped.train_features - center) / scale
    test_scaled = (clipped.test_features - center) / scale
    metadata = dict(clipped.metadata)
    metadata.update(
        {
            "source_clip_standardize": True,
            "source_clip_standardize_protocol": SOURCE_CLIP_STANDARDIZE_PROTOCOL,
            "source_clip_standardize_protocol_category": SOURCE_CLIP_CATEGORY,
            "source_clip_standardize_uses_source_features": True,
            "source_clip_standardize_uses_test_features_for_fitting": False,
            "source_clip_standardize_uses_test_labels": False,
            "source_clip_standardize_valid_for_strict_source_only": True,
            "source_clip_standardize_valid_for_benchmark": True,
            "source_clip_standardize_epsilon": float(eps),
            "source_clip_standardize_min_scale": float(np.min(scale)),
            "source_clip_standardize_max_scale": float(np.max(scale)),
        }
    )
    return SourceClipStandardizeResult(
        train_features=_compact_float32(train_scaled),
        test_features=_compact_float32(test_scaled),
        clip_result=clipped,
        center=center.astype(float, copy=False),
        scale=scale.astype(float, copy=False),
        metadata=metadata,
    )


def source_clip_config(
    *,
    lower_quantile: float | str = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float | str = DEFAULT_UPPER_QUANTILE,
    symmetric: bool | str | int | float = False,
    center: str | None = "median",
) -> SourceClipConfig:
    return SourceClipConfig(lower_quantile=lower_quantile, upper_quantile=upper_quantile, symmetric=symmetric, center=center)


def normalize_center_mode(value: str | None) -> str:
    normalized = "median" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"med": "median", "avg": "mean", "average": "mean", "none": "zero"}.get(normalized, normalized)
    if normalized not in CENTER_MODES:
        raise ValueError(f"Unknown center mode {value!r}. Available values: {', '.join(CENTER_MODES)}.")
    return normalized


def fit_source_clip_bounds(source_features: Sequence[Sequence[float]] | np.ndarray, *, config: SourceClipConfig | Mapping[str, Any] | None = None) -> SourceClipBounds:
    cfg = source_clip_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    center = _center_vector(source, mode=cfg.center)
    if cfg.symmetric:
        radius = _upper_sample_quantile(np.abs(source - center), cfg.upper_quantile)
        lower = center - radius
        upper = center + radius
    else:
        lower = _lower_sample_quantile(source, cfg.lower_quantile)
        upper = _upper_sample_quantile(source, cfg.upper_quantile)
    lower, upper = _repair_bounds(lower, upper)
    return SourceClipBounds(lower=lower.astype(float, copy=False), upper=upper.astype(float, copy=False), center=center.astype(float, copy=False), n_source_rows=int(source.shape[0]))


def apply_source_clip(features: Sequence[Sequence[float]] | np.ndarray, bounds: SourceClipBounds) -> tuple[np.ndarray, np.ndarray]:
    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != bounds.lower.shape[0] or matrix.shape[1] != bounds.upper.shape[0]:
        raise ValueError("features width must match clipping bounds.")
    clipped = np.clip(matrix, bounds.lower, bounds.upper)
    return clipped, clipped != matrix


def _coerce_config(config: SourceClipConfig | Mapping[str, Any]) -> SourceClipConfig:
    if isinstance(config, SourceClipConfig):
        return config
    return source_clip_config(**dict(config))


def _lower_sample_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    return np.quantile(values, quantile, axis=0, method="lower")


def _upper_sample_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    return np.quantile(values, quantile, axis=0, method="higher")


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


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when conversion preserves finite, nonzero values."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return values
    if np.any((values != 0.0) & (compact == 0.0)):
        return values
    return compact


def _metadata(cfg: SourceClipConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, train_mask: np.ndarray, test_mask: np.ndarray) -> dict[str, Any]:
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


def _bool_config(value: bool | str | int | float, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a boolean value.")
        return _bool_config(value.item(), name=name)
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


def _numeric_scalar(value: Any, *, message: str) -> float:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _unit_interval_float(value: Any, *, name: str) -> float:
    message = f"{name} must be in [0, 1]."
    parsed = _numeric_scalar(value, message=message)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(message)
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    message = f"{name} must be positive and finite."
    parsed = _numeric_scalar(value, message=message)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(message)
    return parsed
