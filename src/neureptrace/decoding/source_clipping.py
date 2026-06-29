"""Source-only feature clipping for robust cross-subject decoding.

The transform estimates per-feature lower and upper bounds from source rows only
and applies those bounds to train/test feature matrices.  It is useful as a
strict Protocol-1 preprocessing baseline for reducing outlier sensitivity without
using held-out-domain information.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_CLIPPING_PROTOCOL = "strict_source_only_feature_clipping"
SOURCE_CLIPPING_CATEGORY = "1_strict_source_only"
DEFAULT_LOWER_QUANTILE = 0.01
DEFAULT_UPPER_QUANTILE = 0.99


@dataclass(frozen=True, slots=True)
class SourceFeatureClippingConfig:
    """Configuration for source-only feature clipping."""

    lower_quantile: float = DEFAULT_LOWER_QUANTILE
    upper_quantile: float = DEFAULT_UPPER_QUANTILE
    copy: bool = True


@dataclass(frozen=True, slots=True)
class SourceFeatureClippingResult:
    """Clipped train/test features and fitted source bounds."""

    train_features: np.ndarray
    test_features: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_feature_clipping(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceFeatureClippingConfig | Mapping[str, Any] | None = None,
) -> SourceFeatureClippingResult:
    """Fit clipping bounds on source rows and transform source/test rows."""

    cfg = source_feature_clipping_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    lower, upper = source_feature_clipping_bounds(
        source,
        lower_quantile=cfg.lower_quantile,
        upper_quantile=cfg.upper_quantile,
    )
    train = apply_feature_clipping(source, lower_bounds=lower, upper_bounds=upper, copy=cfg.copy)
    test_clipped = apply_feature_clipping(test, lower_bounds=lower, upper_bounds=upper, copy=cfg.copy)
    metadata = {
        "source_feature_clipping": True,
        "source_feature_clipping_protocol": SOURCE_CLIPPING_PROTOCOL,
        "source_feature_clipping_protocol_category": SOURCE_CLIPPING_CATEGORY,
        "source_feature_clipping_uses_source_features": True,
        "source_feature_clipping_uses_source_labels": False,
        "source_feature_clipping_uses_test_features_for_fitting": False,
        "source_feature_clipping_uses_test_labels": False,
        "source_feature_clipping_valid_for_strict_source_only": True,
        "source_feature_clipping_valid_for_benchmark": True,
        "source_feature_clipping_n_source_rows": int(source.shape[0]),
        "source_feature_clipping_n_test_rows": int(test.shape[0]),
        "source_feature_clipping_feature_dim": int(source.shape[1]),
        "source_feature_clipping_lower_quantile": float(cfg.lower_quantile),
        "source_feature_clipping_upper_quantile": float(cfg.upper_quantile),
    }
    return SourceFeatureClippingResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_clipped.astype(np.float32, copy=False),
        lower_bounds=lower.astype(np.float32, copy=False),
        upper_bounds=upper.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_feature_clipping_config(
    *,
    lower_quantile: float | str = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float | str = DEFAULT_UPPER_QUANTILE,
    copy: bool | int | str = True,
) -> SourceFeatureClippingConfig:
    """Normalize public clipping options."""

    lower = _quantile(lower_quantile, name="lower_quantile")
    upper = _quantile(upper_quantile, name="upper_quantile")
    if lower >= upper:
        raise ValueError("lower_quantile must be smaller than upper_quantile.")
    return SourceFeatureClippingConfig(lower_quantile=lower, upper_quantile=upper, copy=_bool_value(copy, name="copy"))


def source_feature_clipping_bounds(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate feature-wise source clipping bounds."""

    source = _feature_matrix(source_features, name="source_features")
    lower = _quantile(lower_quantile, name="lower_quantile")
    upper = _quantile(upper_quantile, name="upper_quantile")
    if lower >= upper:
        raise ValueError("lower_quantile must be smaller than upper_quantile.")
    lower_bounds = np.quantile(source, lower, axis=0)
    upper_bounds = np.quantile(source, upper, axis=0)
    return lower_bounds.astype(float, copy=False), upper_bounds.astype(float, copy=False)


def apply_feature_clipping(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    lower_bounds: Sequence[float] | np.ndarray,
    upper_bounds: Sequence[float] | np.ndarray,
    copy: bool | int | str = True,
) -> np.ndarray:
    """Apply precomputed clipping bounds to feature rows."""

    matrix = _feature_matrix(features, name="features")
    lower = np.asarray(lower_bounds, dtype=float).reshape(-1)
    upper = np.asarray(upper_bounds, dtype=float).reshape(-1)
    if lower.shape[0] != matrix.shape[1] or upper.shape[0] != matrix.shape[1]:
        raise ValueError("lower_bounds and upper_bounds must match the feature width.")
    if np.any(lower > upper):
        raise ValueError("lower_bounds cannot exceed upper_bounds.")
    target = matrix.copy() if _bool_value(copy, name="copy") else matrix
    return np.clip(target, lower, upper, out=target)


def _coerce_config(config: SourceFeatureClippingConfig | Mapping[str, Any]) -> SourceFeatureClippingConfig:
    if isinstance(config, SourceFeatureClippingConfig):
        return config
    return source_feature_clipping_config(**dict(config))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _quantile(value: float | str, *, name: str) -> float:
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
