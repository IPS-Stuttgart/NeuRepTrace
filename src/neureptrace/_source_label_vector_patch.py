"""Runtime patch for source-decoder label vector normalization."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_DANN_ROW_ERROR = "DANN source_features and source_labels must contain the same rows."
_DANN_SHAPE_ERROR = "DANN source_labels must contain one label per source row."
_SOURCE_DG_ROW_ERROR = "source_features and source_labels must contain the same rows."
_SOURCE_DG_SHAPE_ERROR = "source-domain generalization source_labels must contain one label per source row."
_SOURCE_PROTOTYPE_CLASS_PATCH_MARKER = "_source_prototype_explicit_class_vector_patched"


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


def _as_class_vector(classes: Any) -> np.ndarray:
    """Return explicit class labels without flattening composite row labels."""

    if isinstance(classes, (str, bytes)):
        return _object_value_vector([classes])

    class_array = np.asarray(classes, dtype=object)
    if class_array.ndim == 0:
        return _object_value_vector([class_array.item()])
    if class_array.ndim == 1:
        return class_array.reshape(-1)
    if class_array.shape[0] == 0:
        return _object_value_vector([])

    flattened_width = int(np.prod(class_array.shape[1:], dtype=np.int64))
    if flattened_width == 1:
        return class_array.reshape(class_array.shape[0])

    rows = class_array.reshape(class_array.shape[0], flattened_width)
    return _object_value_vector(tuple(row.tolist()) for row in rows)


def _install_source_prototype_class_patch() -> None:
    source_prototypes = importlib.import_module("neureptrace.decoding.source_prototype_features")
    if getattr(source_prototypes.class_prototypes, _SOURCE_PROTOTYPE_CLASS_PATCH_MARKER, False):
        return

    original_class_prototypes = source_prototypes.class_prototypes

    @wraps(original_class_prototypes)
    def class_prototypes(source_features: Any, source_labels: Any, *, classes: Any = None) -> np.ndarray:
        if classes is not None:
            classes = _as_class_vector(classes)
        return original_class_prototypes(source_features, source_labels, classes=classes)

    setattr(class_prototypes, _SOURCE_PROTOTYPE_CLASS_PATCH_MARKER, True)
    source_prototypes.class_prototypes = class_prototypes


def install() -> None:
    """Install source-label vector normalization for neural source decoders."""
    source_centroid_patch = importlib.import_module("neureptrace._source_centroid_numeric_config_patch")
    source_centroid_patch.install()
    _install_source_prototype_class_patch()

    import neureptrace.decoding.dann as dann
    import neureptrace.decoding.source_domain_generalization as source_dg

    if not getattr(dann.TorchDANNClassifier.fit, "_source_label_vector_patched", False):
        original_dann_fit = dann.TorchDANNClassifier.fit

        @wraps(original_dann_fit)
        def fit_dann(self, source_features: np.ndarray, source_labels: np.ndarray, *, target_features: np.ndarray):
            n_rows = _source_row_count(source_features)
            if n_rows is not None:
                source_labels = _as_label_vector(source_labels, n_rows=n_rows, row_error=_DANN_ROW_ERROR, shape_error=_DANN_SHAPE_ERROR)
            return original_dann_fit(self, source_features, source_labels, target_features=target_features)

        fit_dann._source_label_vector_patched = True  # type: ignore[attr-defined]
        dann.TorchDANNClassifier.fit = fit_dann

    if not getattr(source_dg.TorchSourceDomainGeneralizationClassifier.fit, "_source_label_vector_patched", False):
        original_source_dg_fit = source_dg.TorchSourceDomainGeneralizationClassifier.fit

        @wraps(original_source_dg_fit)
        def fit_source_dg(self, source_features: np.ndarray, source_labels: np.ndarray, *, source_domains: np.ndarray):
            n_rows = _source_row_count(source_features)
            if n_rows is not None:
                source_labels = _as_label_vector(source_labels, n_rows=n_rows, row_error=_SOURCE_DG_ROW_ERROR, shape_error=_SOURCE_DG_SHAPE_ERROR)
            return original_source_dg_fit(self, source_features, source_labels, source_domains=source_domains)

        fit_source_dg._source_label_vector_patched = True  # type: ignore[attr-defined]
        source_dg.TorchSourceDomainGeneralizationClassifier.fit = fit_source_dg


__all__ = ["install"]
