"""Strict source-only ZCA whitening helpers.

This module fits a ZCA whitening transform from source rows only and applies the
fixed transform to source and held-out matrices.  It is a Protocol-1 preprocessing
helper: held-out rows are transformed, but never used to estimate the mean,
covariance, or whitening matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_ZCA_PROTOCOL = "strict_source_only_zca_whitening"
SOURCE_ZCA_CATEGORY = "1_strict_source_only"
DEFAULT_REGULARIZATION = 1e-5
_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True, slots=True)
class SourceZCAConfig:
    """Configuration for source-only ZCA whitening."""

    regularization: float = DEFAULT_REGULARIZATION
    center: bool = True
    recolor: bool = False

    def __post_init__(self) -> None:
        """Normalize and validate direct dataclass construction."""

        object.__setattr__(self, "regularization", _positive_float(self.regularization, name="regularization"))
        object.__setattr__(self, "center", _bool_value(self.center, name="center"))
        object.__setattr__(self, "recolor", _bool_value(self.recolor, name="recolor"))


@dataclass(frozen=True, slots=True)
class SourceZCAReference:
    """A source-fitted ZCA whitening reference."""

    mean: np.ndarray
    whitening: np.ndarray
    coloring: np.ndarray
    eigenvalues: np.ndarray
    config: SourceZCAConfig
    n_fit_rows: int


@dataclass(frozen=True, slots=True)
class SourceZCAResult:
    """Whitened source/test rows and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourceZCAReference
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_zca_transform(
    *,
    source_features: Iterable[Iterable[float]] | np.ndarray,
    test_features: Iterable[Iterable[float]] | np.ndarray,
    config: SourceZCAConfig | Mapping[str, Any] | None = None,
) -> SourceZCAResult:
    """Fit source-only ZCA whitening and transform source/test rows."""

    cfg = source_zca_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    reference = fit_source_zca_reference(source, config=cfg)
    train = apply_source_zca_transform(source, reference)
    test_out = apply_source_zca_transform(test, reference)
    metadata = _metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1])
    return SourceZCAResult(train_features=train, test_features=test_out, reference=reference, metadata=metadata)


def fit_source_zca_reference(source_features: Iterable[Iterable[float]] | np.ndarray, *, config: SourceZCAConfig | Mapping[str, Any] | None = None) -> SourceZCAReference:
    """Fit a ZCA reference from source rows only."""

    cfg = source_zca_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    mean = np.mean(source, axis=0) if cfg.center else np.zeros(source.shape[1], dtype=float)
    centered = source - mean
    covariance = _covariance(centered)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0.0)
    whitening = (vectors * (1.0 / np.sqrt(values + cfg.regularization))) @ vectors.T
    coloring = (vectors * np.sqrt(values + cfg.regularization)) @ vectors.T
    return SourceZCAReference(
        mean=mean.astype(float, copy=False),
        whitening=_float32_if_safe(whitening),
        coloring=_float32_if_safe(coloring),
        eigenvalues=values.astype(float, copy=False),
        config=cfg,
        n_fit_rows=int(source.shape[0]),
    )


def apply_source_zca_transform(features: Iterable[Iterable[float]] | np.ndarray, reference: SourceZCAReference) -> np.ndarray:
    """Apply a fitted source-only ZCA transform."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != reference.whitening.shape[0]:
        raise ValueError(f"features width {matrix.shape[1]} does not match ZCA reference width {reference.whitening.shape[0]}.")
    transformed = (matrix - reference.mean) @ reference.whitening
    if reference.config.recolor:
        transformed = transformed @ reference.coloring
    return _float32_if_safe(transformed)


def source_zca_config(*, regularization: Any = DEFAULT_REGULARIZATION, center: Any = True, recolor: Any = False) -> SourceZCAConfig:
    """Normalize source-ZCA options."""

    return SourceZCAConfig(regularization=_positive_float(regularization, name="regularization"), center=_bool_value(center, name="center"), recolor=_bool_value(recolor, name="recolor"))


def _coerce_config(config: SourceZCAConfig | Mapping[str, Any]) -> SourceZCAConfig:
    if isinstance(config, SourceZCAConfig):
        return source_zca_config(
            regularization=config.regularization,
            center=config.center,
            recolor=config.recolor,
        )
    return source_zca_config(**dict(config))


def _covariance(centered: np.ndarray) -> np.ndarray:
    if centered.shape[0] <= 1:
        return np.zeros((centered.shape[1], centered.shape[1]), dtype=float)
    return centered.T @ centered / float(centered.shape[0] - 1)


def _float32_if_safe(values: np.ndarray) -> np.ndarray:
    """Use float32 when representable, otherwise retain finite float64 values."""

    array = np.asarray(values, dtype=float)
    if array.size and bool(np.any(np.abs(array) > _FLOAT32_MAX)):
        return array
    return array.astype(np.float32, copy=False)


def _metadata(cfg: SourceZCAConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int) -> dict[str, Any]:
    return {
        "source_zca": True,
        "source_zca_protocol": SOURCE_ZCA_PROTOCOL,
        "source_zca_protocol_category": SOURCE_ZCA_CATEGORY,
        "source_zca_uses_source_features": True,
        "source_zca_uses_source_labels": False,
        "source_zca_uses_test_features_for_fitting": False,
        "source_zca_uses_test_labels": False,
        "source_zca_valid_for_strict_source_only": True,
        "source_zca_valid_for_benchmark": True,
        "source_zca_n_source_rows": int(n_source_rows),
        "source_zca_n_test_rows": int(n_test_rows),
        "source_zca_feature_dim": int(feature_dim),
        "source_zca_regularization": float(cfg.regularization),
        "source_zca_center": bool(cfg.center),
        "source_zca_recolor": bool(cfg.recolor),
    }


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize nested one-pass iterables before NumPy consumes them."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_one_pass_iterables(value.tolist())
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
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if isinstance(value, Iterable):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _feature_matrix(values: Iterable[Iterable[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _positive_float(value: Any, *, name: str) -> float:
    message = f"{name} must be a positive finite scalar."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or np.issubdtype(value.dtype, np.bool_):
            raise ValueError(message)
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(message)
    return parsed


def _bool_value(value: Any, *, name: str) -> bool:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a boolean value.")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
        if parsed in {0, 1}:
            return bool(parsed)
    if isinstance(value, (float, np.floating)):
        parsed_float = float(value)
        if np.isfinite(parsed_float) and parsed_float in {0.0, 1.0}:
            return bool(parsed_float)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")
