"""Strict source-only variance feature selection.

This module selects feature columns using source-row statistics only and then
applies the fixed selection to source and held-out matrices.  It is a small
Protocol-1 preprocessing helper for fold-local decoding workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_FEATURE_SELECT_PROTOCOL = "strict_source_only_variance_feature_selection"
SOURCE_FEATURE_SELECT_CATEGORY = "1_strict_source_only"


@dataclass(frozen=True, slots=True)
class SourceFeatureSelectResult:
    """Selected source/test feature matrices and provenance."""

    train_features: np.ndarray
    test_features: np.ndarray
    selected_indices: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def select_source_variance_features(
    *,
    source_features,
    test_features,
    k: int | str | None = None,
    min_variance: float | str | None = None,
) -> SourceFeatureSelectResult:
    """Select source-variance-ranked feature columns and apply to test rows.

    ``source_features`` are used to fit the selected feature indices.  ``test``
    rows are transformed with the fixed indices and are not used for fitting.
    """

    source = _matrix(source_features, name="source_features")
    test = _matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError("source_features and test_features must have the same feature width.")
    scores = _stable_column_variances(source)
    parsed_k = None if k is None else _positive_int(k, name="k")
    parsed_min_variance = None if min_variance is None else _nonnegative_float(min_variance, name="min_variance")
    selected = source_variance_feature_indices(scores=scores, k=parsed_k, min_variance=parsed_min_variance)
    metadata = {
        "source_feature_select": True,
        "source_feature_select_protocol": SOURCE_FEATURE_SELECT_PROTOCOL,
        "source_feature_select_protocol_category": SOURCE_FEATURE_SELECT_CATEGORY,
        "source_feature_select_uses_source_features": True,
        "source_feature_select_uses_test_features_for_fitting": False,
        "source_feature_select_uses_test_labels": False,
        "source_feature_select_valid_for_strict_source_only": True,
        "source_feature_select_valid_for_benchmark": True,
        "source_feature_select_n_source_rows": int(source.shape[0]),
        "source_feature_select_n_test_rows": int(test.shape[0]),
        "source_feature_select_feature_dim": int(source.shape[1]),
        "source_feature_select_n_selected_features": int(selected.shape[0]),
        "source_feature_select_k": "" if parsed_k is None else parsed_k,
        "source_feature_select_min_variance": "" if parsed_min_variance is None else parsed_min_variance,
    }
    return SourceFeatureSelectResult(
        train_features=_compact_float32(source[:, selected]),
        test_features=_compact_float32(test[:, selected]),
        selected_indices=selected.astype(int, copy=False),
        scores=_compact_float32(scores),
        metadata=metadata,
    )


def source_variance_feature_indices(*, scores, k: int | str | None = None, min_variance: float | str | None = None) -> np.ndarray:
    """Return selected feature indices from source-only variance scores."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.shape[0] < 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("scores must contain finite non-negative values.")
    mask = np.ones(values.shape[0], dtype=bool)
    if min_variance is not None:
        threshold = _nonnegative_float(min_variance, name="min_variance")
        mask &= values >= threshold
    candidate = np.flatnonzero(mask)
    if candidate.size == 0:
        candidate = np.asarray([int(np.argmax(values))], dtype=int)
    if k is not None:
        count = _positive_int(k, name="k")
        count = min(count, candidate.shape[0])
        order = np.argsort(values[candidate], kind="mergesort")[::-1][:count]
        candidate = candidate[order]
    else:
        candidate = candidate[np.argsort(values[candidate], kind="mergesort")[::-1]]
    return np.sort(candidate).astype(int, copy=False)


def _stable_column_variances(matrix: np.ndarray) -> np.ndarray:
    """Compute finite column variances without overflowing intermediate moments."""

    magnitude = np.max(np.abs(matrix), axis=0)
    normalized = np.zeros_like(matrix)
    nonzero = magnitude > 0.0
    normalized[:, nonzero] = matrix[:, nonzero] / magnitude[nonzero]
    ddof = 1 if matrix.shape[0] > 1 else 0
    normalized_variance = np.var(normalized, axis=0, ddof=ddof)

    variances = np.zeros_like(magnitude)
    positive = nonzero & (normalized_variance > 0.0)
    if np.any(positive):
        normalized_positive = normalized_variance[positive]
        magnitude_positive = magnitude[positive]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            maximum_magnitude = np.sqrt(np.finfo(float).max / normalized_positive)
            scaled = normalized_positive * magnitude_positive * magnitude_positive
        scaled[magnitude_positive > maximum_magnitude] = np.finfo(float).max
        variances[positive] = scaled
    return variances


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when conversion preserves finite, nonzero values."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return values
    if np.any((values != 0.0) & (compact == 0.0)):
        return values
    return compact


def _matrix(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _finite_scalar_float(value, *, name: str) -> float:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if array.shape != ():
        raise ValueError(f"{name} must be a finite scalar.")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar.")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite scalar.")
    return parsed


def _nonnegative_float(value, *, name: str) -> float:
    parsed = _finite_scalar_float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _positive_int(value, *, name: str) -> int:
    parsed = _finite_scalar_float(value, name=name)
    if parsed < 1.0 or not parsed.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)
