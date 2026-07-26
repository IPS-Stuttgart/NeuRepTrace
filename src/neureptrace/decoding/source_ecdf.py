"""Strict source-only empirical-CDF feature transform.

This module fits feature-wise empirical CDF references from source rows only and
applies the fixed transform to source and held-out rows.  It can return uniform
scores, normal scores, or rank counts.  Held-out rows are transformed but never
used to fit the empirical reference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

SOURCE_ECDF_PROTOCOL = "strict_source_only_empirical_cdf_transform"
SOURCE_ECDF_CATEGORY = "1_strict_source_only"
OUTPUT_MODES = ("uniform", "normal", "rank")
DEFAULT_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class SourceECDFConfig:
    """Configuration for source-only empirical-CDF transformation."""

    output: str = "uniform"
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        """Normalize and validate direct dataclass construction."""

        object.__setattr__(self, "output", normalize_ecdf_output(self.output))
        object.__setattr__(self, "epsilon", _unit_interval_open_float(self.epsilon, name="epsilon"))


@dataclass(frozen=True, slots=True)
class SourceECDFReference:
    """Source-fitted empirical-CDF reference."""

    sorted_values: np.ndarray
    output: str
    epsilon: float
    n_source_rows: int


@dataclass(frozen=True, slots=True)
class SourceECDFResult:
    """Transformed source/test rows and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourceECDFReference
    metadata: dict[str, Any] = field(default_factory=dict)


# Keep the explicit keyword-only API; callers use named inputs in configs.
# pylint: disable-next=too-many-arguments,too-many-locals
def fit_source_ecdf_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceECDFConfig | Mapping[str, Any] | None = None,
) -> SourceECDFResult:
    """Fit source ECDF references and transform source/test rows."""

    cfg = source_ecdf_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    reference = fit_source_ecdf_reference(source, config=cfg)
    train = apply_source_ecdf_transform(source, reference)
    test_out = apply_source_ecdf_transform(test, reference)
    return SourceECDFResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        reference=reference,
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1]),
    )


def source_ecdf_config(*, output: str | None = "uniform", epsilon: float | str = DEFAULT_EPSILON) -> SourceECDFConfig:
    """Normalize public ECDF transform options."""

    return SourceECDFConfig(output=normalize_ecdf_output(output), epsilon=_unit_interval_open_float(epsilon, name="epsilon"))


def normalize_ecdf_output(value: str | None) -> str:
    """Normalize output-mode aliases."""

    normalized = "uniform" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"cdf": "uniform", "probability": "uniform", "normal_score": "normal", "gaussian": "normal", "count": "rank"}.get(normalized, normalized)
    if normalized not in OUTPUT_MODES:
        raise ValueError(f"Unknown ECDF output mode {value!r}. Available values: {', '.join(OUTPUT_MODES)}.")
    return normalized


def fit_source_ecdf_reference(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceECDFConfig | Mapping[str, Any] | None = None,
) -> SourceECDFReference:
    """Fit feature-wise sorted source values."""

    cfg = source_ecdf_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    sorted_values = np.sort(source, axis=0).T
    return SourceECDFReference(sorted_values=sorted_values.astype(float, copy=False), output=cfg.output, epsilon=cfg.epsilon, n_source_rows=int(source.shape[0]))


def apply_source_ecdf_transform(features: Sequence[Sequence[float]] | np.ndarray, reference: SourceECDFReference) -> np.ndarray:
    """Apply a source-fitted ECDF reference to feature rows."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != reference.sorted_values.shape[0]:
        raise ValueError(
            "features width must match ECDF reference width: "
            f"{matrix.shape[1]} != {reference.sorted_values.shape[0]}."
        )
    ranks = np.empty_like(matrix, dtype=float)
    constant_columns: list[int] = []
    for feature_index in range(matrix.shape[1]):
        sorted_feature = reference.sorted_values[feature_index]
        ranks[:, feature_index] = np.searchsorted(sorted_feature, matrix[:, feature_index], side="right")
        if sorted_feature[0] == sorted_feature[-1]:
            constant_columns.append(feature_index)
    if reference.output == "rank":
        return ranks
    uniform = (ranks + 0.5) / float(reference.n_source_rows + 1)
    uniform = np.clip(uniform, reference.epsilon, 1.0 - reference.epsilon)
    if constant_columns:
        uniform[:, constant_columns] = 0.5
    if reference.output == "uniform":
        return uniform
    if reference.output == "normal":
        return _normal_scores(uniform)
    raise ValueError(f"Unhandled ECDF output mode {reference.output!r}.")


def _normal_scores(uniform: np.ndarray) -> np.ndarray:
    normal = NormalDist()
    flat = uniform.reshape(-1)
    values = np.asarray([normal.inv_cdf(float(value)) for value in flat], dtype=float)
    return values.reshape(uniform.shape)


def _coerce_config(config: SourceECDFConfig | Mapping[str, Any]) -> SourceECDFConfig:
    if isinstance(config, SourceECDFConfig):
        return source_ecdf_config(output=config.output, epsilon=config.epsilon)
    return source_ecdf_config(**dict(config))


def _metadata(cfg: SourceECDFConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int) -> dict[str, Any]:
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
        "source_ecdf_output": cfg.output,
        "source_ecdf_epsilon": float(cfg.epsilon),
    }


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize nested one-pass iterables before NumPy coercion."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_one_pass_iterables(value.tolist())
    if isinstance(value, (str, bytes)):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_one_pass_iterables(item) for item in value]


def _contains_complex(value: object) -> bool:
    """Return whether a materialized feature input contains complex values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Iterable):
        return False
    return any(_contains_complex(item) for item in value)


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(values)
    if _contains_complex(materialized):
        raise ValueError(f"{name} must contain only real-valued features.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _unit_interval_open_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in (0, 0.5).") from exc
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed >= 0.5:
        raise ValueError(f"{name} must be in (0, 0.5).")
    return parsed
