"""Source-only quantile helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCE_QUANTILE_CATEGORY = "1_strict_source_only"
SOURCE_QUANTILE_CLIP_PROTOCOL = "strict_source_only_quantile_clip"
SOURCE_QUANTILE_RANK_PROTOCOL = "strict_source_only_quantile_rank"
SOURCE_QUANTILE_BIN_PROTOCOL = "strict_source_only_quantile_bins"


@dataclass(frozen=True, slots=True)
class SourceQuantileClipResult:
    """Feature matrices clipped with source-fitted quantiles."""

    train_features: np.ndarray
    test_features: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    train_clipped_mask: np.ndarray
    test_clipped_mask: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class SourceQuantileRankResult:
    """Feature matrices transformed with source-fitted empirical ranks."""

    train_features: np.ndarray
    test_features: np.ndarray
    sorted_source_values: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class SourceQuantileBinResult:
    """Feature matrices transformed to source-fitted quantile bin indices."""

    train_features: np.ndarray
    test_features: np.ndarray
    bin_edges: np.ndarray
    metadata: dict[str, object]


def source_feature_quantiles(source_features, *, lower=0.01, upper=0.99):
    matrix = _matrix(source_features, name="source_features")
    lo, hi = _bounds(lower, upper)
    return np.quantile(matrix, lo, axis=0), np.quantile(matrix, hi, axis=0)


def source_quantile_clip(*, source_features, test_features, lower=0.01, upper=0.99) -> SourceQuantileClipResult:
    """Clip source and test rows using quantiles fitted on source rows only."""

    source = _matrix(source_features, name="source_features")
    test = _matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError("source_features and test_features must have the same feature width.")
    lo, hi = _bounds(lower, upper)
    lower_values, upper_values = source_feature_quantiles(source, lower=lo, upper=hi)
    train, train_mask = apply_source_quantile_clip(source, lower_values, upper_values)
    test_out, test_mask = apply_source_quantile_clip(test, lower_values, upper_values)
    metadata = {
        "source_quantile_clip": True,
        "source_quantile_clip_protocol": SOURCE_QUANTILE_CLIP_PROTOCOL,
        "source_quantile_clip_protocol_category": SOURCE_QUANTILE_CATEGORY,
        "source_quantile_clip_uses_source_features": True,
        "source_quantile_clip_uses_test_features_for_fitting": False,
        "source_quantile_clip_uses_test_labels": False,
        "source_quantile_clip_valid_for_strict_source_only": True,
        "source_quantile_clip_n_source_rows": int(source.shape[0]),
        "source_quantile_clip_n_test_rows": int(test.shape[0]),
        "source_quantile_clip_feature_dim": int(source.shape[1]),
        "source_quantile_clip_lower": float(lo),
        "source_quantile_clip_upper": float(hi),
        "source_quantile_clip_train_values_clipped": int(np.count_nonzero(train_mask)),
        "source_quantile_clip_test_values_clipped": int(np.count_nonzero(test_mask)),
    }
    return SourceQuantileClipResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        lower=lower_values.astype(float, copy=False),
        upper=upper_values.astype(float, copy=False),
        train_clipped_mask=train_mask,
        test_clipped_mask=test_mask,
        metadata=metadata,
    )


def source_quantile_rank(*, source_features, test_features, centered: bool = False, epsilon=1e-6) -> SourceQuantileRankResult:
    """Transform rows to empirical CDF/rank features fitted on source rows only.

    ``centered=False`` returns values in ``[epsilon, 1 - epsilon]``.  When
    ``centered=True``, the same source-fitted ranks are mapped to ``[-1, 1]``.
    """

    source = _matrix(source_features, name="source_features")
    test = _matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError("source_features and test_features must have the same feature width.")
    eps = _epsilon(epsilon)
    sorted_values = np.sort(source, axis=0)
    train = apply_source_quantile_rank(source, sorted_values=sorted_values, centered=centered, epsilon=eps)
    test_out = apply_source_quantile_rank(test, sorted_values=sorted_values, centered=centered, epsilon=eps)
    metadata = {
        "source_quantile_rank": True,
        "source_quantile_rank_protocol": SOURCE_QUANTILE_RANK_PROTOCOL,
        "source_quantile_rank_protocol_category": SOURCE_QUANTILE_CATEGORY,
        "source_quantile_rank_uses_source_features": True,
        "source_quantile_rank_uses_test_features_for_fitting": False,
        "source_quantile_rank_uses_test_labels": False,
        "source_quantile_rank_valid_for_strict_source_only": True,
        "source_quantile_rank_n_source_rows": int(source.shape[0]),
        "source_quantile_rank_n_test_rows": int(test.shape[0]),
        "source_quantile_rank_feature_dim": int(source.shape[1]),
        "source_quantile_rank_centered": bool(centered),
        "source_quantile_rank_epsilon": float(eps),
    }
    return SourceQuantileRankResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        sorted_source_values=sorted_values.astype(float, copy=False),
        metadata=metadata,
    )


def source_quantile_bins(*, source_features, test_features, n_bins=4) -> SourceQuantileBinResult:
    """Transform rows to source-fitted quantile bin indices.

    The bin edges are fitted from source rows only.  Returned values are integer
    bin indices in ``[0, n_bins - 1]`` for each feature column.
    """

    source = _matrix(source_features, name="source_features")
    test = _matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError("source_features and test_features must have the same feature width.")
    bins = _positive_int(n_bins, name="n_bins")
    levels = np.linspace(0.0, 1.0, bins + 1, dtype=float)[1:-1]
    edges = np.quantile(source, levels, axis=0) if levels.size else np.empty((0, source.shape[1]), dtype=float)
    train = apply_source_quantile_bins(source, bin_edges=edges)
    test_out = apply_source_quantile_bins(test, bin_edges=edges)
    metadata = {
        "source_quantile_bins": True,
        "source_quantile_bins_protocol": SOURCE_QUANTILE_BIN_PROTOCOL,
        "source_quantile_bins_protocol_category": SOURCE_QUANTILE_CATEGORY,
        "source_quantile_bins_uses_source_features": True,
        "source_quantile_bins_uses_test_features_for_fitting": False,
        "source_quantile_bins_uses_test_labels": False,
        "source_quantile_bins_valid_for_strict_source_only": True,
        "source_quantile_bins_n_source_rows": int(source.shape[0]),
        "source_quantile_bins_n_test_rows": int(test.shape[0]),
        "source_quantile_bins_feature_dim": int(source.shape[1]),
        "source_quantile_bins_n_bins": int(bins),
    }
    return SourceQuantileBinResult(
        train_features=train,
        test_features=test_out,
        bin_edges=edges.astype(float, copy=False),
        metadata=metadata,
    )


def apply_source_quantile_rank(features, *, sorted_values, centered: bool = False, epsilon=1e-6):
    matrix = _matrix(features, name="features")
    reference = _matrix(sorted_values, name="sorted_values")
    if matrix.shape[1] != reference.shape[1]:
        raise ValueError("features width must match sorted source values.")
    eps = _epsilon(epsilon)
    n_rows = reference.shape[0]
    ranks = np.empty_like(matrix, dtype=float)
    for column in range(matrix.shape[1]):
        values = reference[:, column]
        left = np.searchsorted(values, matrix[:, column], side="left")
        right = np.searchsorted(values, matrix[:, column], side="right")
        ranks[:, column] = (left + right) / (2.0 * n_rows)
    ranks = np.clip(ranks, eps, 1.0 - eps)
    if centered:
        return 2.0 * ranks - 1.0
    return ranks


def apply_source_quantile_bins(features, *, bin_edges):
    matrix = _matrix(features, name="features")
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 2:
        raise ValueError("bin_edges must be a two-dimensional matrix.")
    if matrix.shape[1] != edges.shape[1]:
        raise ValueError("features width must match source quantile bin edges.")
    output = np.zeros(matrix.shape, dtype=np.int16)
    for column in range(matrix.shape[1]):
        output[:, column] = np.searchsorted(edges[:, column], matrix[:, column], side="right").astype(np.int16, copy=False)
    return output


def apply_source_quantile_clip(features, lower, upper):
    matrix = _matrix(features, name="features")
    lower_values = np.asarray(lower, dtype=float).reshape(-1)
    upper_values = np.asarray(upper, dtype=float).reshape(-1)
    if matrix.shape[1] != lower_values.shape[0] or matrix.shape[1] != upper_values.shape[0]:
        raise ValueError("features width must match lower and upper bounds.")
    clipped = np.minimum(np.maximum(matrix, lower_values), upper_values)
    return clipped, clipped != matrix


def _matrix(values, *, name: str):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _bounds(lower, upper) -> tuple[float, float]:
    if isinstance(lower, (bool, np.bool_)) or isinstance(upper, (bool, np.bool_)):
        raise ValueError("lower and upper must be numeric quantiles, not boolean.")
    try:
        lo = float(lower)
        hi = float(upper)
    except (TypeError, ValueError) as exc:
        raise ValueError("lower and upper must satisfy 0 <= lower <= upper <= 1.") from exc
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError("lower and upper must satisfy 0 <= lower <= upper <= 1.")
    return lo, hi


def _epsilon(epsilon) -> float:
    value = float(epsilon)
    if not np.isfinite(value) or value <= 0.0 or value >= 0.5:
        raise ValueError("epsilon must be finite and in (0, 0.5).")
    return value


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)
