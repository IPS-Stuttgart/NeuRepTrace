"""Source-only feature clipping for robust cross-subject decoding.

The transform estimates per-feature lower and upper bounds from source rows only
and applies those bounds to train/test feature matrices.  It is useful as a
strict Protocol-1 preprocessing baseline for reducing outlier sensitivity without
using held-out-domain information.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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

    def __post_init__(self) -> None:
        lower = _quantile(self.lower_quantile, name="lower_quantile")
        upper = _quantile(self.upper_quantile, name="upper_quantile")
        if lower >= upper:
            raise ValueError("lower_quantile must be smaller than upper_quantile.")
        object.__setattr__(self, "lower_quantile", lower)
        object.__setattr__(self, "upper_quantile", upper)
        object.__setattr__(self, "copy", _bool_value(self.copy, name="copy"))


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
        train_features=_compact_float32(train),
        test_features=_compact_float32(test_clipped),
        lower_bounds=_compact_float32(lower),
        upper_bounds=_compact_float32(upper),
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
    lower = _bound_vector(lower_bounds, name="lower_bounds")
    upper = _bound_vector(upper_bounds, name="upper_bounds")
    if lower.shape[0] != matrix.shape[1] or upper.shape[0] != matrix.shape[1]:
        raise ValueError("lower_bounds and upper_bounds must match the feature width.")
    if np.any(lower > upper):
        raise ValueError("lower_bounds cannot exceed upper_bounds.")
    target = matrix.copy() if _bool_value(copy, name="copy") else matrix
    return np.clip(target, lower, upper, out=target)


def _coerce_config(config: SourceFeatureClippingConfig | Mapping[str, Any]) -> SourceFeatureClippingConfig:
    if isinstance(config, SourceFeatureClippingConfig):
        return source_feature_clipping_config(
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            copy=config.copy,
        )
    return source_feature_clipping_config(**dict(config))


def _contains_complex(value: object) -> bool:
    """Return whether an input container contains complex values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, np.generic):
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Mapping):
        return any(_contains_complex(item) for item in value.values())
    if isinstance(value, Iterable):
        return any(_contains_complex(item) for item in value)
    return False


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    if _contains_complex(values):
        raise ValueError(f"{name} must contain real-valued feature values, not complex values.")
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _bound_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    """Return one numeric clipping-bound vector without lossy coercion or NaNs."""

    if _contains_complex(values):
        raise ValueError(f"{name} must contain real-valued bounds.")
    try:
        vector = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector.") from exc
    if np.any(np.isnan(vector)):
        raise ValueError(f"{name} must not contain NaN values.")
    return vector


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when conversion preserves finite, nonzero values."""

    array = np.asarray(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return array
    if np.any((array != 0.0) & (compact == 0.0)):
        return array
    return compact


def _scalar_array_value(value: object, *, name: str) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar value.")
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quantile(value: float | str, *, name: str) -> float:
    value = _scalar_array_value(value, name=name)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric quantile, not boolean.")
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f"{name} must be a real numeric scalar quantile, not complex.")
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
