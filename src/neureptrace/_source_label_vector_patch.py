"""Runtime patch for source-decoder label vector normalization."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_DANN_ROW_ERROR = "DANN source_features and source_labels must contain the same rows."
_DANN_SHAPE_ERROR = "DANN source_labels must contain one label per source row."
_SOURCE_DG_ROW_ERROR = "source_features and source_labels must contain the same rows."
_SOURCE_DG_SHAPE_ERROR = "source-domain generalization source_labels must contain one label per source row."


def _source_row_count(source_features: Any) -> int | None:
    features = np.asarray(source_features)
    if features.ndim == 0:
        return None
    return int(features.shape[0])


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _as_label_vector(source_labels: Any, *, n_rows: int, row_error: str, shape_error: str) -> np.ndarray:
    labels = np.asarray(source_labels, dtype=object)
    if labels.ndim == 0 or labels.shape[0] != n_rows:
        raise ValueError(row_error)
    if labels.ndim == 1:
        return labels.reshape(n_rows)

    flattened_width = int(np.prod(labels.shape[1:], dtype=np.int64))
    if flattened_width == 1:
        return labels.reshape(n_rows)
    if flattened_width < 1:
        raise ValueError(shape_error)

    rows = labels.reshape(n_rows, flattened_width)
    return _object_value_vector(tuple(row.tolist()) for row in rows)


def install() -> None:
    """Install source-label vector normalization for neural source decoders."""
    import neureptrace.decoding.dann as dann
    import neureptrace.decoding.source_domain_generalization as source_dg

    if not getattr(dann.TorchDANNClassifier.fit, "_source_label_vector_patched", False):
        original_dann_fit = dann.TorchDANNClassifier.fit

        @wraps(original_dann_fit)
        def fit_dann(self, source_features: np.ndarray, source_labels: np.ndarray, *, target_features: np.ndarray):
            n_rows = _source_row_count(source_features)
            if n_rows is not None:
                source_labels = _as_label_vector(
                    source_labels,
                    n_rows=n_rows,
                    row_error=_DANN_ROW_ERROR,
                    shape_error=_DANN_SHAPE_ERROR,
                )
            return original_dann_fit(self, source_features, source_labels, target_features=target_features)

        fit_dann._source_label_vector_patched = True  # type: ignore[attr-defined]
        dann.TorchDANNClassifier.fit = fit_dann

    if not getattr(source_dg.TorchSourceDomainGeneralizationClassifier.fit, "_source_label_vector_patched", False):
        original_source_dg_fit = source_dg.TorchSourceDomainGeneralizationClassifier.fit

        @wraps(original_source_dg_fit)
        def fit_source_dg(self, source_features: np.ndarray, source_labels: np.ndarray, *, source_domains: np.ndarray):
            n_rows = _source_row_count(source_features)
            if n_rows is not None:
                source_labels = _as_label_vector(
                    source_labels,
                    n_rows=n_rows,
                    row_error=_SOURCE_DG_ROW_ERROR,
                    shape_error=_SOURCE_DG_SHAPE_ERROR,
                )
            return original_source_dg_fit(self, source_features, source_labels, source_domains=source_domains)

        fit_source_dg._source_label_vector_patched = True  # type: ignore[attr-defined]
        source_dg.TorchSourceDomainGeneralizationClassifier.fit = fit_source_dg


__all__ = ["install"]
