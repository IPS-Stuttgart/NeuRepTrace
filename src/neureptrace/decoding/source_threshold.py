"""Strict source-only threshold transforms.

This module fits feature-wise thresholds from source rows only and applies the
fixed threshold map to source and held-out rows.  It can emit binary indicators or
signed values and is intended as a small fold-local preprocessing baseline for
cross-subject decoding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_THRESHOLD_PROTOCOL = "strict_source_only_threshold_transform"
SOURCE_THRESHOLD_CATEGORY = "1_strict_source_only"
THRESHOLD_MODES = ("median", "mean", "quantile", "zero")
OUTPUT_MODES = ("binary", "signed")
DEFAULT_QUANTILE = 0.5


@dataclass(frozen=True, slots=True)
class SourceThresholdConfig:
    """Configuration for source-fitted threshold transforms."""

    threshold_mode: str = "median"
    quantile: float = DEFAULT_QUANTILE
    output: str = "binary"
    positive_value: float = 1.0
    negative_value: float = 0.0

    def __post_init__(self) -> None:
        """Normalize and validate direct dataclass construction."""

        object.__setattr__(self, "threshold_mode", normalize_threshold_mode(self.threshold_mode))
        object.__setattr__(self, "quantile", _unit_interval_float(self.quantile, name="quantile"))
        object.__setattr__(self, "output", normalize_output_mode(self.output))
        object.__setattr__(self, "positive_value", _finite_float(self.positive_value, name="positive_value"))
        object.__setattr__(self, "negative_value", _finite_float(self.negative_value, name="negative_value"))


@dataclass(frozen=True, slots=True)
class SourceThresholdMap:
    """Feature-wise thresholds fitted from source rows."""

    thresholds: np.ndarray
    threshold_mode: str
    output: str
    positive_value: float
    negative_value: float
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceThresholdResult:
    """Transformed train/test rows plus fitted map and metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    threshold_map: SourceThresholdMap
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_threshold_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceThresholdConfig | Mapping[str, Any] | None = None,
) -> SourceThresholdResult:
    """Fit source-only thresholds and transform source/test rows."""

    cfg = source_threshold_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    threshold_map = fit_source_threshold_map(source, config=cfg)
    train = apply_source_threshold_transform(source, threshold_map)
    test_out = apply_source_threshold_transform(test, threshold_map)
    return SourceThresholdResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        threshold_map=threshold_map,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1]),
    )


def source_threshold_config(
    *,
    threshold_mode: str | None = "median",
    quantile: float | str = DEFAULT_QUANTILE,
    output: str | None = "binary",
    positive_value: float | str = 1.0,
    negative_value: float | str = 0.0,
) -> SourceThresholdConfig:
    """Normalize source-threshold options."""

    return SourceThresholdConfig(
        threshold_mode=threshold_mode,
        quantile=quantile,
        output=output,
        positive_value=positive_value,
        negative_value=negative_value,
    )


def normalize_threshold_mode(value: str | None) -> str:
    """Normalize threshold mode aliases."""

    normalized = "median" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"med": "median", "avg": "mean", "average": "mean", "q": "quantile", "constant_zero": "zero"}.get(normalized, normalized)
    if normalized not in THRESHOLD_MODES:
        raise ValueError(f"Unknown threshold_mode {value!r}. Available values: {', '.join(THRESHOLD_MODES)}.")
    return normalized


def normalize_output_mode(value: str | None) -> str:
    """Normalize output mode aliases."""

    normalized = "binary" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"bool": "binary", "indicator": "binary", "sign": "signed", "pm1": "signed"}.get(normalized, normalized)
    if normalized not in OUTPUT_MODES:
        raise ValueError(f"Unknown output {value!r}. Available values: {', '.join(OUTPUT_MODES)}.")
    return normalized


def fit_source_threshold_map(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceThresholdConfig | Mapping[str, Any] | None = None,
) -> SourceThresholdMap:
    """Estimate feature-wise thresholds from source rows only."""

    cfg = source_threshold_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    if cfg.threshold_mode == "median":
        thresholds = np.median(source, axis=0)
    elif cfg.threshold_mode == "mean":
        thresholds = np.mean(source, axis=0)
    elif cfg.threshold_mode == "quantile":
        thresholds = np.quantile(source, cfg.quantile, axis=0)
    elif cfg.threshold_mode == "zero":
        thresholds = np.zeros(source.shape[1], dtype=float)
    else:  # pragma: no cover - normalized above
        raise ValueError(f"Unhandled threshold_mode {cfg.threshold_mode!r}.")
    return SourceThresholdMap(
        thresholds=thresholds.astype(float, copy=False),
        threshold_mode=cfg.threshold_mode,
        output=cfg.output,
        positive_value=float(cfg.positive_value),
        negative_value=float(cfg.negative_value),
        n_source_rows=int(source.shape[0]),
    )


def apply_source_threshold_transform(features: Sequence[Sequence[float]] | np.ndarray, threshold_map: SourceThresholdMap) -> np.ndarray:
    """Apply source-fitted thresholds to feature rows."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != threshold_map.thresholds.shape[0]:
        raise ValueError("features width must match threshold map width.")
    high = matrix >= threshold_map.thresholds[None, :]
    if threshold_map.output == "binary":
        return np.where(high, threshold_map.positive_value, threshold_map.negative_value)
    if threshold_map.output == "signed":
        return np.where(high, abs(threshold_map.positive_value), -abs(threshold_map.positive_value))
    raise ValueError(f"Unhandled output mode {threshold_map.output!r}.")


def _coerce_config(config: SourceThresholdConfig | Mapping[str, Any]) -> SourceThresholdConfig:
    if isinstance(config, SourceThresholdConfig):
        return config
    return source_threshold_config(**dict(config))


def _metadata(cfg: SourceThresholdConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_threshold_transform": True,
        "source_threshold_protocol": SOURCE_THRESHOLD_PROTOCOL,
        "source_threshold_protocol_category": SOURCE_THRESHOLD_CATEGORY,
        "source_threshold_uses_source_features": True,
        "source_threshold_uses_test_features_for_fitting": False,
        "source_threshold_uses_labels": False,
        "source_threshold_valid_for_strict_source_only": True,
        "source_threshold_valid_for_benchmark": True,
        "source_threshold_n_source_rows": int(n_source_rows),
        "source_threshold_n_test_rows": int(n_test_rows),
        "source_threshold_feature_dim": int(feature_dim),
        "source_threshold_mode": cfg.threshold_mode,
        "source_threshold_quantile": float(cfg.quantile),
        "source_threshold_output": cfg.output,
        "source_threshold_positive_value": float(cfg.positive_value),
        "source_threshold_negative_value": float(cfg.negative_value),
    }


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize nested one-pass iterables before NumPy consumes them."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_one_pass_iterables(value.tolist())
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Mapping):
        return {key: _materialize_one_pass_iterables(item) for key, item in value.items()}
    if not isinstance(value, Iterable):
        return value
    return [_materialize_one_pass_iterables(item) for item in value]


def _contains_boolean(value: object) -> bool:
    """Return whether a materialized feature container contains boolean values."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.flat)
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Mapping):
        return any(_contains_boolean(item) for item in value.values())
    if isinstance(value, Iterable):
        return any(_contains_boolean(item) for item in value)
    return False


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean(materialized):
        raise ValueError(f"{name} must contain numeric, non-boolean feature values.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _finite_float(value: float | str, *, name: str) -> float:
    message = f"{name} must be finite."
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or np.issubdtype(value.dtype, np.bool_):
            raise ValueError(message)
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_, list, tuple, dict, set)):
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed):
        raise ValueError(message)
    return parsed
