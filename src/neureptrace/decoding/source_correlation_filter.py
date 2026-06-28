"""Source-only correlation feature filtering.

This module fits a feature-redundancy mask from source rows only.  Features are
ranked by a source-only importance vector, then greedily kept when their absolute
source correlation with previously kept features does not exceed the configured
threshold.  Evaluation rows are transformed with the fitted mask but never used to
fit it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_CORRELATION_FILTER_PROTOCOL = "strict_source_only_correlation_feature_filter"
SOURCE_CORRELATION_FILTER_CATEGORY = "1_strict_source_only"
DEFAULT_MAX_ABS_CORRELATION = 0.98
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceCorrelationFilterConfig:
    """Configuration for source-only correlation feature filtering."""

    max_abs_correlation: float = DEFAULT_MAX_ABS_CORRELATION
    max_features: int | None = None
    min_features: int = 1
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceCorrelationFilterResult:
    """Filtered feature matrices and fitted source-only mask."""

    train_features: np.ndarray
    test_features: np.ndarray
    selected_indices: np.ndarray
    correlation: np.ndarray
    importance: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-locals

def fit_source_correlation_filter(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceCorrelationFilterConfig | Mapping[str, Any] | None = None,
) -> SourceCorrelationFilterResult:
    """Fit a source-only correlation mask and transform feature matrices."""

    cfg = source_correlation_filter_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    correlation = source_feature_correlation(source, epsilon=cfg.epsilon)
    importance = source_feature_importance(source, epsilon=cfg.epsilon)
    selected = select_uncorrelated_features(
        correlation,
        importance=importance,
        max_abs_correlation=cfg.max_abs_correlation,
        max_features=cfg.max_features,
        min_features=cfg.min_features,
    )
    metadata = {
        "source_correlation_filter": True,
        "source_correlation_filter_protocol": SOURCE_CORRELATION_FILTER_PROTOCOL,
        "source_correlation_filter_protocol_category": SOURCE_CORRELATION_FILTER_CATEGORY,
        "source_correlation_filter_uses_source_features": True,
        "source_correlation_filter_uses_source_labels": False,
        "source_correlation_filter_uses_test_features_for_fitting": False,
        "source_correlation_filter_uses_test_labels": False,
        "source_correlation_filter_valid_for_strict_source_only": True,
        "source_correlation_filter_valid_for_benchmark": True,
        "source_correlation_filter_n_source_rows": int(source.shape[0]),
        "source_correlation_filter_n_test_rows": int(test.shape[0]),
        "source_correlation_filter_input_dim": int(source.shape[1]),
        "source_correlation_filter_output_dim": int(selected.shape[0]),
        "source_correlation_filter_max_abs_correlation": float(cfg.max_abs_correlation),
        "source_correlation_filter_max_features": "" if cfg.max_features is None else int(cfg.max_features),
        "source_correlation_filter_min_features": int(cfg.min_features),
        "source_correlation_filter_selected_indices": "|".join(str(int(index)) for index in selected.tolist()),
    }
    return SourceCorrelationFilterResult(
        train_features=source[:, selected].astype(np.float32, copy=False),
        test_features=test[:, selected].astype(np.float32, copy=False),
        selected_indices=selected.astype(int, copy=False),
        correlation=correlation.astype(np.float32, copy=False),
        importance=importance.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_correlation_filter_config(
    *,
    max_abs_correlation: float | str = DEFAULT_MAX_ABS_CORRELATION,
    max_features: int | str | None = None,
    min_features: int | str = 1,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceCorrelationFilterConfig:
    """Normalize public source-correlation-filter options."""

    return SourceCorrelationFilterConfig(
        max_abs_correlation=_unit_interval_float(max_abs_correlation, name="max_abs_correlation"),
        max_features=None if max_features in {None, "", "none", "None"} else _positive_int(max_features, name="max_features"),
        min_features=_positive_int(min_features, name="min_features"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def source_feature_correlation(source_features: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Return the source-only feature correlation matrix."""

    source = _feature_matrix(source_features, name="source_features")
    eps = _positive_float(epsilon, name="epsilon")
    centered = source - np.mean(source, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    safe_norms = np.maximum(norms, eps)
    normalized = centered / safe_norms[None, :]
    correlation = normalized.T @ normalized
    zero_var = norms <= eps
    if np.any(zero_var):
        correlation[zero_var, :] = 0.0
        correlation[:, zero_var] = 0.0
        correlation[zero_var, zero_var] = 1.0
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation.astype(float, copy=False)


def source_feature_importance(source_features: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Return source-only feature importance based on variance."""

    source = _feature_matrix(source_features, name="source_features")
    variances = np.var(source, axis=0, ddof=1 if source.shape[0] > 1 else 0)
    return np.maximum(variances, _positive_float(epsilon, name="epsilon")).astype(float, copy=False)


def select_uncorrelated_features(
    correlation: Sequence[Sequence[float]] | np.ndarray,
    *,
    importance: Sequence[float] | np.ndarray | None = None,
    max_abs_correlation: float = DEFAULT_MAX_ABS_CORRELATION,
    max_features: int | None = None,
    min_features: int = 1,
) -> np.ndarray:
    """Greedily select low-redundancy feature indices."""

    corr = np.asarray(correlation, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] < 1:
        raise ValueError("correlation must be a non-empty square matrix.")
    if not np.all(np.isfinite(corr)):
        raise ValueError("correlation must contain only finite values.")
    threshold = _unit_interval_float(max_abs_correlation, name="max_abs_correlation")
    min_keep = _positive_int(min_features, name="min_features")
    if min_keep > corr.shape[0]:
        raise ValueError("min_features cannot exceed the number of features.")
    max_keep = corr.shape[0] if max_features is None else min(_positive_int(max_features, name="max_features"), corr.shape[0])
    if max_keep < min_keep:
        raise ValueError("max_features must be greater than or equal to min_features.")
    scores = np.ones(corr.shape[0], dtype=float) if importance is None else np.asarray(importance, dtype=float).reshape(-1)
    if scores.shape[0] != corr.shape[0] or not np.all(np.isfinite(scores)):
        raise ValueError("importance must contain one finite score per feature.")
    order = np.lexsort((np.arange(scores.shape[0]), -scores))
    selected: list[int] = []
    for candidate in order.tolist():
        if len(selected) >= max_keep:
            break
        if not selected or np.all(np.abs(corr[candidate, selected]) <= threshold):
            selected.append(int(candidate))
    if len(selected) < min_keep:
        for candidate in order.tolist():
            if candidate not in selected:
                selected.append(int(candidate))
                if len(selected) >= min_keep:
                    break
    return np.sort(np.asarray(selected, dtype=int))


def _coerce_config(config: SourceCorrelationFilterConfig | Mapping[str, Any]) -> SourceCorrelationFilterConfig:
    if isinstance(config, SourceCorrelationFilterConfig):
        return config
    return source_correlation_filter_config(**dict(config))


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


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed
