"""Source-only variance feature filtering.

This module fits a feature-selection mask from source rows only and applies that
mask to source and evaluation feature matrices.  It is a strict Protocol-1
preprocessing helper for removing constant or low-variance features without using
any evaluation-domain statistics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_VARIANCE_FILTER_PROTOCOL = "strict_source_only_variance_feature_filter"
SOURCE_VARIANCE_FILTER_CATEGORY = "1_strict_source_only"
DEFAULT_VARIANCE_THRESHOLD = 0.0


@dataclass(frozen=True, slots=True)
class SourceVarianceFilterConfig:
    """Configuration for source-only variance feature filtering."""

    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    top_k: int | None = None
    ddof: int = 1


@dataclass(frozen=True, slots=True)
class SourceVarianceFilterResult:
    """Filtered feature matrices and fitted source-only mask."""

    train_features: np.ndarray
    test_features: np.ndarray
    selected_indices: np.ndarray
    variances: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-locals


def fit_source_variance_filter(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceVarianceFilterConfig | Mapping[str, Any] | None = None,
) -> SourceVarianceFilterResult:
    """Fit a variance-based feature mask from source rows and transform matrices."""

    cfg = source_variance_filter_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    variances = source_feature_variances(source, ddof=cfg.ddof)
    selected = select_variance_features(variances, variance_threshold=cfg.variance_threshold, top_k=cfg.top_k)
    metadata = {
        "source_variance_filter": True,
        "source_variance_filter_protocol": SOURCE_VARIANCE_FILTER_PROTOCOL,
        "source_variance_filter_protocol_category": SOURCE_VARIANCE_FILTER_CATEGORY,
        "source_variance_filter_uses_source_features": True,
        "source_variance_filter_uses_source_labels": False,
        "source_variance_filter_uses_test_features_for_fitting": False,
        "source_variance_filter_uses_test_labels": False,
        "source_variance_filter_valid_for_strict_source_only": True,
        "source_variance_filter_valid_for_benchmark": True,
        "source_variance_filter_n_source_rows": int(source.shape[0]),
        "source_variance_filter_n_test_rows": int(test.shape[0]),
        "source_variance_filter_input_dim": int(source.shape[1]),
        "source_variance_filter_output_dim": int(selected.shape[0]),
        "source_variance_filter_variance_threshold": float(cfg.variance_threshold),
        "source_variance_filter_top_k": "" if cfg.top_k is None else int(cfg.top_k),
        "source_variance_filter_ddof": int(cfg.ddof),
        "source_variance_filter_selected_indices": "|".join(str(int(index)) for index in selected.tolist()),
    }
    return SourceVarianceFilterResult(
        train_features=source[:, selected].astype(np.float32, copy=False),
        test_features=test[:, selected].astype(np.float32, copy=False),
        selected_indices=selected.astype(int, copy=False),
        variances=variances.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_variance_filter_config(
    *,
    variance_threshold: float | str = DEFAULT_VARIANCE_THRESHOLD,
    top_k: int | str | None = None,
    ddof: int | str = 1,
) -> SourceVarianceFilterConfig:
    """Normalize public variance-filter options."""

    return SourceVarianceFilterConfig(
        variance_threshold=_nonnegative_float(variance_threshold, name="variance_threshold"),
        top_k=None if top_k in {None, "", "none", "None"} else _positive_int(top_k, name="top_k"),
        ddof=_nonnegative_int(ddof, name="ddof"),
    )


def source_feature_variances(source_features: Sequence[Sequence[float]] | np.ndarray, *, ddof: int = 1) -> np.ndarray:
    """Return source-only feature variances."""

    source = _feature_matrix(source_features, name="source_features")
    resolved_ddof = _nonnegative_int(ddof, name="ddof")
    if source.shape[0] <= resolved_ddof:
        resolved_ddof = 0
    return np.var(source, axis=0, ddof=resolved_ddof).astype(float, copy=False)


def select_variance_features(
    variances: Sequence[float] | np.ndarray,
    *,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    top_k: int | None = None,
) -> np.ndarray:
    """Return selected feature indices sorted in original feature order."""

    values = np.asarray(variances, dtype=float).reshape(-1)
    if values.size < 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("variances must be a non-empty finite non-negative vector.")
    threshold = _nonnegative_float(variance_threshold, name="variance_threshold")
    selected = np.flatnonzero(values > threshold)
    if top_k is not None:
        k = min(_positive_int(top_k, name="top_k"), values.size)
        ranked = np.argsort(values, kind="mergesort")[-k:]
        selected = np.intersect1d(selected, ranked, assume_unique=False)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(values))], dtype=int)
    return np.sort(selected).astype(int, copy=False)


def _coerce_config(config: SourceVarianceFilterConfig | Mapping[str, Any]) -> SourceVarianceFilterConfig:
    if isinstance(config, SourceVarianceFilterConfig):
        return config
    return source_variance_filter_config(**dict(config))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _nonnegative_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)


def _nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be non-negative and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be non-negative and finite.") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed
