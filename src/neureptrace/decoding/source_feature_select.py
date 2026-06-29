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
    scores = np.var(source - np.mean(source, axis=0), axis=0, ddof=1 if source.shape[0] > 1 else 0)
    selected = source_variance_feature_indices(scores=scores, k=k, min_variance=min_variance)
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
        "source_feature_select_k": "" if k is None else int(k),
        "source_feature_select_min_variance": "" if min_variance is None else float(min_variance),
    }
    return SourceFeatureSelectResult(
        train_features=source[:, selected].astype(np.float32, copy=False),
        test_features=test[:, selected].astype(np.float32, copy=False),
        selected_indices=selected.astype(int, copy=False),
        scores=scores.astype(np.float32, copy=False),
        metadata=metadata,
    )


def source_variance_feature_indices(*, scores, k: int | str | None = None, min_variance: float | str | None = None) -> np.ndarray:
    """Return selected feature indices from source-only variance scores."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.shape[0] < 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("scores must contain finite non-negative values.")
    mask = np.ones(values.shape[0], dtype=bool)
    if min_variance is not None:
        threshold = float(min_variance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("min_variance must be finite and non-negative.")
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


def _matrix(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _positive_int(value, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)
