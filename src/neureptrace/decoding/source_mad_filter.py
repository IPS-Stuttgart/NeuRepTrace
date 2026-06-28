"""Source-only median-absolute-deviation feature filtering.

The selected feature mask is fitted from source rows only, using per-feature
median absolute deviation (MAD). Evaluation rows are transformed with that fitted
mask but never used to fit it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_MAD_FILTER_PROTOCOL = "strict_source_only_mad_feature_filter"
SOURCE_MAD_FILTER_CATEGORY = "1_strict_source_only"
DEFAULT_MAD_THRESHOLD = 0.0


@dataclass(frozen=True, slots=True)
class SourceMadFilterConfig:
    """Configuration for source-only MAD feature filtering."""

    mad_threshold: float = DEFAULT_MAD_THRESHOLD
    top_k: int | None = None
    scale_to_sigma: bool = True


@dataclass(frozen=True, slots=True)
class SourceMadFilterResult:
    """Filtered feature matrices and fitted source-only mask."""

    train_features: np.ndarray
    test_features: np.ndarray
    selected_indices: np.ndarray
    mad: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_mad_filter(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceMadFilterConfig | Mapping[str, Any] | None = None,
) -> SourceMadFilterResult:
    """Fit a source-only MAD feature mask and transform matrices."""

    cfg = source_mad_filter_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    mad = source_feature_mad(source, scale_to_sigma=cfg.scale_to_sigma)
    selected = select_mad_features(mad, mad_threshold=cfg.mad_threshold, top_k=cfg.top_k)
    metadata = {
        "source_mad_filter": True,
        "source_mad_filter_protocol": SOURCE_MAD_FILTER_PROTOCOL,
        "source_mad_filter_protocol_category": SOURCE_MAD_FILTER_CATEGORY,
        "source_mad_filter_uses_source_features": True,
        "source_mad_filter_uses_source_labels": False,
        "source_mad_filter_uses_test_features_for_fitting": False,
        "source_mad_filter_uses_test_labels": False,
        "source_mad_filter_valid_for_strict_source_only": True,
        "source_mad_filter_valid_for_benchmark": True,
        "source_mad_filter_n_source_rows": int(source.shape[0]),
        "source_mad_filter_n_test_rows": int(test.shape[0]),
        "source_mad_filter_input_dim": int(source.shape[1]),
        "source_mad_filter_output_dim": int(selected.shape[0]),
        "source_mad_filter_mad_threshold": float(cfg.mad_threshold),
        "source_mad_filter_top_k": "" if cfg.top_k is None else int(cfg.top_k),
        "source_mad_filter_scale_to_sigma": bool(cfg.scale_to_sigma),
        "source_mad_filter_selected_indices": "|".join(str(int(index)) for index in selected.tolist()),
    }
    return SourceMadFilterResult(
        train_features=source[:, selected].astype(np.float32, copy=False),
        test_features=test[:, selected].astype(np.float32, copy=False),
        selected_indices=selected.astype(int, copy=False),
        mad=mad.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_mad_filter_config(
    *,
    mad_threshold: float | str = DEFAULT_MAD_THRESHOLD,
    top_k: int | str | None = None,
    scale_to_sigma: bool = True,
) -> SourceMadFilterConfig:
    """Normalize public MAD-filter options."""

    return SourceMadFilterConfig(
        mad_threshold=_nonnegative_float(mad_threshold, name="mad_threshold"),
        top_k=None if top_k in {None, "", "none", "None"} else _positive_int(top_k, name="top_k"),
        scale_to_sigma=bool(scale_to_sigma),
    )


def source_feature_mad(source_features: Sequence[Sequence[float]] | np.ndarray, *, scale_to_sigma: bool = True) -> np.ndarray:
    """Return feature-wise source median absolute deviations."""

    source = _feature_matrix(source_features, name="source_features")
    med = np.median(source, axis=0)
    mad = np.median(np.abs(source - med), axis=0)
    if scale_to_sigma:
        mad = mad * 1.4826
    return mad.astype(float, copy=False)


def select_mad_features(mad: Sequence[float] | np.ndarray, *, mad_threshold: float = DEFAULT_MAD_THRESHOLD, top_k: int | None = None) -> np.ndarray:
    """Return selected feature indices sorted in original feature order."""

    values = np.asarray(mad, dtype=float).reshape(-1)
    if values.size < 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("mad must be a non-empty finite non-negative vector.")
    threshold = _nonnegative_float(mad_threshold, name="mad_threshold")
    selected = np.flatnonzero(values > threshold)
    if top_k is not None:
        k = min(_positive_int(top_k, name="top_k"), values.size)
        ranked = np.argsort(values, kind="mergesort")[-k:]
        selected = np.intersect1d(selected, ranked, assume_unique=False)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(values))], dtype=int)
    return np.sort(selected).astype(int, copy=False)


def _coerce_config(config: SourceMadFilterConfig | Mapping[str, Any]) -> SourceMadFilterConfig:
    if isinstance(config, SourceMadFilterConfig):
        return config
    return source_mad_filter_config(**dict(config))


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


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed
