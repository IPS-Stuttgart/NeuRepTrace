"""Strict source-only inverse-hyperbolic-sine feature transform.

This module fits optional feature scales from source rows only and then applies a
signed asinh compression to source and held-out rows.  It is useful as a small
fold-local preprocessing baseline when feature magnitudes are heavy-tailed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_ASINH_PROTOCOL = "strict_source_only_asinh_transform"
SOURCE_ASINH_CATEGORY = "1_strict_source_only"
SCALE_MODES = ("unit", "std", "mad", "iqr")
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class SourceAsinhConfig:
    """Configuration for source-fitted asinh compression."""

    scale_mode: str = "mad"
    multiplier: float = 1.0
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        """Normalize direct dataclass construction consistently with the config helper."""

        object.__setattr__(self, "scale_mode", normalize_scale_mode(self.scale_mode))
        object.__setattr__(self, "multiplier", _positive_float(self.multiplier, name="multiplier"))
        object.__setattr__(self, "epsilon", _positive_float(self.epsilon, name="epsilon"))


@dataclass(frozen=True, slots=True)
class SourceAsinhMap:
    """Feature-wise scale map fitted from source rows."""

    scale: np.ndarray
    scale_mode: str
    multiplier: float
    epsilon: float
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceAsinhResult:
    """Transformed source/test features plus fitted map and metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    transform_map: SourceAsinhMap
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_asinh_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceAsinhConfig | Mapping[str, Any] | None = None,
) -> SourceAsinhResult:
    """Fit source-only scales and transform source/test rows with asinh."""

    cfg = source_asinh_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    transform_map = fit_source_asinh_map(source, config=cfg)
    train = apply_source_asinh_transform(source, transform_map)
    test_out = apply_source_asinh_transform(test, transform_map)
    return SourceAsinhResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        transform_map=transform_map,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1]),
    )


def source_asinh_config(
    *,
    scale_mode: str | None = "mad",
    multiplier: float | str = 1.0,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceAsinhConfig:
    """Normalize public asinh transform options."""

    return SourceAsinhConfig(
        scale_mode=normalize_scale_mode(scale_mode),
        multiplier=_positive_float(multiplier, name="multiplier"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_scale_mode(value: str | None) -> str:
    """Normalize scale-mode aliases."""

    normalized = "mad" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "none": "unit",
        "identity": "unit",
        "standard_deviation": "std",
        "sd": "std",
        "median_absolute_deviation": "mad",
        "robust": "mad",
        "interquartile": "iqr",
    }.get(normalized, normalized)
    if normalized not in SCALE_MODES:
        raise ValueError(f"Unknown scale_mode {value!r}. Available values: {', '.join(SCALE_MODES)}.")
    return normalized


def fit_source_asinh_map(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceAsinhConfig | Mapping[str, Any] | None = None,
) -> SourceAsinhMap:
    """Estimate feature-wise source scales for asinh compression."""

    cfg = source_asinh_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    scale = _feature_scale(source, mode=cfg.scale_mode, epsilon=cfg.epsilon) * cfg.multiplier
    scale = np.maximum(scale, cfg.epsilon)
    return SourceAsinhMap(
        scale=scale.astype(float, copy=False),
        scale_mode=cfg.scale_mode,
        multiplier=float(cfg.multiplier),
        epsilon=float(cfg.epsilon),
        n_source_rows=int(source.shape[0]),
    )


def apply_source_asinh_transform(features: Sequence[Sequence[float]] | np.ndarray, transform_map: SourceAsinhMap) -> np.ndarray:
    """Apply a source-fitted asinh compression map."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != transform_map.scale.shape[0]:
        raise ValueError("features width must match asinh transform width.")
    return np.arcsinh(matrix / transform_map.scale[None, :])


def _coerce_config(config: SourceAsinhConfig | Mapping[str, Any]) -> SourceAsinhConfig:
    if isinstance(config, SourceAsinhConfig):
        return config
    return source_asinh_config(**dict(config))


def _feature_scale(source: np.ndarray, *, mode: str, epsilon: float) -> np.ndarray:
    if mode == "unit":
        return np.ones(source.shape[1], dtype=float)
    if mode == "std":
        return np.maximum(np.std(source - np.mean(source, axis=0), axis=0, ddof=1 if source.shape[0] > 1 else 0), epsilon)
    if mode == "mad":
        median = np.median(source, axis=0)
        return np.maximum(1.4826 * np.median(np.abs(source - median), axis=0), epsilon)
    if mode == "iqr":
        q75 = np.percentile(source, 75.0, axis=0)
        q25 = np.percentile(source, 25.0, axis=0)
        return np.maximum((q75 - q25) / 1.349, epsilon)
    raise ValueError(f"Unhandled scale mode {mode!r}.")


def _metadata(cfg: SourceAsinhConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_asinh_transform": True,
        "source_asinh_protocol": SOURCE_ASINH_PROTOCOL,
        "source_asinh_protocol_category": SOURCE_ASINH_CATEGORY,
        "source_asinh_uses_source_features": True,
        "source_asinh_uses_test_features_for_fitting": False,
        "source_asinh_uses_labels": False,
        "source_asinh_valid_for_strict_source_only": True,
        "source_asinh_valid_for_benchmark": True,
        "source_asinh_n_source_rows": int(n_source_rows),
        "source_asinh_n_test_rows": int(n_test_rows),
        "source_asinh_feature_dim": int(feature_dim),
        "source_asinh_scale_mode": cfg.scale_mode,
        "source_asinh_multiplier": float(cfg.multiplier),
        "source_asinh_epsilon": float(cfg.epsilon),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _positive_float(value: float | str, *, name: str) -> float:
    message = f"{name} must be positive and finite."
    scalar = _scalar_numeric_value(value, message=message)
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    try:
        parsed = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(message)
    return parsed


def _scalar_numeric_value(value: Any, *, message: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        return value.item()
    return value
