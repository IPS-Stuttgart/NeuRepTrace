"""Strict source-only empirical CDF transform.

Fits feature-wise empirical CDF breakpoints from source rows only and applies the
fixed mapping to source and held-out rows.  Held-out rows are transformed but not
used to estimate the mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_ECDF_PROTOCOL = "strict_source_only_ecdf_transform"
SOURCE_ECDF_CATEGORY = "1_strict_source_only"
DEFAULT_N_QUANTILES = 128
DEFAULT_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class SourceEcdfConfig:
    """Configuration for source-fitted empirical CDF transforms."""

    n_quantiles: int = DEFAULT_N_QUANTILES
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceEcdfMap:
    """Feature-wise empirical source CDF map."""

    references: np.ndarray
    quantiles: np.ndarray
    epsilon: float
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceEcdfResult:
    """Transformed source/test features and provenance."""

    train_features: np.ndarray
    test_features: np.ndarray
    ecdf_map: SourceEcdfMap
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_ecdf_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceEcdfConfig | Mapping[str, Any] | None = None,
) -> SourceEcdfResult:
    """Fit source empirical CDFs and transform source/test rows."""

    cfg = source_ecdf_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    ecdf_map = fit_source_ecdf_map(source, config=cfg)
    train = apply_source_ecdf_transform(source, ecdf_map)
    test_out = apply_source_ecdf_transform(test, ecdf_map)
    return SourceEcdfResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        ecdf_map=ecdf_map,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], n_quantiles=ecdf_map.references.shape[0]),
    )


def source_ecdf_config(*, n_quantiles: int | str = DEFAULT_N_QUANTILES, epsilon: float | str = DEFAULT_EPSILON) -> SourceEcdfConfig:
    """Normalize public ECDF options."""

    return SourceEcdfConfig(n_quantiles=_positive_int(n_quantiles, name="n_quantiles"), epsilon=_open_unit_float(epsilon, name="epsilon"))


def fit_source_ecdf_map(source_features: Sequence[Sequence[float]] | np.ndarray, *, config: SourceEcdfConfig | Mapping[str, Any] | None = None) -> SourceEcdfMap:
    """Estimate feature-wise source ECDF quantiles."""

    cfg = source_ecdf_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    n_quantiles = min(cfg.n_quantiles, source.shape[0])
    references = np.linspace(0.0, 1.0, n_quantiles, dtype=float)
    quantiles = np.quantile(source, references, axis=0)
    return SourceEcdfMap(references=references, quantiles=quantiles, epsilon=float(cfg.epsilon), n_source_rows=int(source.shape[0]))


def apply_source_ecdf_transform(features: Sequence[Sequence[float]] | np.ndarray, ecdf_map: SourceEcdfMap) -> np.ndarray:
    """Apply a source-fitted ECDF map."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != ecdf_map.quantiles.shape[1]:
        raise ValueError("features width must match ECDF map width.")
    transformed = np.empty_like(matrix, dtype=float)
    for column in range(matrix.shape[1]):
        transformed[:, column] = _interp_unique(matrix[:, column], ecdf_map.quantiles[:, column], ecdf_map.references)
    return np.clip(transformed, ecdf_map.epsilon, 1.0 - ecdf_map.epsilon)


def _coerce_config(config: SourceEcdfConfig | Mapping[str, Any]) -> SourceEcdfConfig:
    if isinstance(config, SourceEcdfConfig):
        return config
    return source_ecdf_config(**dict(config))


def _interp_unique(values: np.ndarray, knots: np.ndarray, references: np.ndarray) -> np.ndarray:
    unique_knots, unique_indices = np.unique(knots, return_index=True)
    unique_refs = references[unique_indices]
    if unique_knots.shape[0] == 1:
        return np.full(values.shape[0], 0.5, dtype=float)
    return np.interp(values, unique_knots, unique_refs, left=0.0, right=1.0)


def _metadata(cfg: SourceEcdfConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, n_quantiles: int) -> dict[str, Any]:
    return {
        "source_ecdf_transform": True,
        "source_ecdf_protocol": SOURCE_ECDF_PROTOCOL,
        "source_ecdf_protocol_category": SOURCE_ECDF_CATEGORY,
        "source_ecdf_uses_source_features": True,
        "source_ecdf_uses_test_features_for_fitting": False,
        "source_ecdf_uses_test_labels": False,
        "source_ecdf_valid_for_strict_source_only": True,
        "source_ecdf_valid_for_benchmark": True,
        "source_ecdf_n_source_rows": int(n_source_rows),
        "source_ecdf_n_test_rows": int(n_test_rows),
        "source_ecdf_feature_dim": int(feature_dim),
        "source_ecdf_n_quantiles": int(n_quantiles),
        "source_ecdf_requested_quantiles": int(cfg.n_quantiles),
        "source_ecdf_epsilon": float(cfg.epsilon),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _positive_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _open_unit_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed >= 0.5:
        raise ValueError(f"{name} must be in (0, 0.5).")
    return parsed
