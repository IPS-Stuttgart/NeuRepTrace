"""Handle NaN/composite labels and complex inputs in source-outlier weighting."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask, values_equal

_PATCH_MARKER = "_neureptrace_source_outlier_nan_labels_patch_installed"
_FEATURE_MATRIX_PATCH_MARKER = "_neureptrace_source_outlier_complex_features_patch_installed"
_NORMALIZE_FLOAT_PATCH_MARKER = "_neureptrace_source_outlier_complex_controls_patch_installed"


class _LabelThresholdMapping(Mapping[Any, float]):
    """Small mapping that looks up label thresholds with object-label equality."""

    def __init__(self, labels: Sequence[Any], values: Sequence[float]) -> None:
        self._labels = tuple(labels)
        self._values = tuple(float(value) for value in values)
        if len(self._labels) != len(self._values):
            raise ValueError("labels and values must have the same length.")

    def __iter__(self) -> Iterator[Any]:
        return iter(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, key: Any) -> float:
        for label, value in zip(self._labels, self._values, strict=True):
            if values_equal(label, key):
                return value
        raise KeyError(key)


def _unique_label_vector(labels: np.ndarray) -> np.ndarray:
    """Return stable unique labels using NaN-aware object-label equality."""

    classes, _counts = label_counts(labels)
    return classes


def _label_equal_mask(labels: np.ndarray, label: Any) -> np.ndarray:
    """Return a mask using the package-wide object-label equality semantics."""

    return label_equal_mask(labels, label)


def _contains_complex_feature_value(value: Any) -> bool:
    """Return whether a materialized feature container contains complex values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return True
        if value.dtype == object:
            return any(_contains_complex_feature_value(item) for item in value.reshape(-1).tolist())
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return any(_contains_complex_feature_value(item) for item in value)
    return False


def install() -> None:
    """Install source-outlier input and source-label compatibility fixes."""

    import neureptrace.decoding.source_outlier as source_outlier

    original_feature_matrix = source_outlier._feature_matrix
    if not getattr(original_feature_matrix, _FEATURE_MATRIX_PATCH_MARKER, False):

        @wraps(original_feature_matrix)
        def _feature_matrix(values: Any, *, name: str) -> np.ndarray:
            materialized_values = source_outlier._materialize_feature_values(values)
            if _contains_complex_feature_value(materialized_values):
                raise ValueError(f"{name} must contain real-valued features.")
            return original_feature_matrix(materialized_values, name=name)

        setattr(_feature_matrix, _FEATURE_MATRIX_PATCH_MARKER, True)
        source_outlier._feature_matrix = _feature_matrix

    original_normalize_float = source_outlier._normalize_float
    if not getattr(original_normalize_float, _NORMALIZE_FLOAT_PATCH_MARKER, False):

        @wraps(original_normalize_float)
        def _normalize_float(value: Any, *, name: str) -> float:
            scalar = value
            if isinstance(scalar, np.ndarray) and scalar.ndim == 0:
                scalar = scalar.item()
            if isinstance(scalar, (complex, np.complexfloating)):
                raise ValueError(f"{name} must be finite and real-valued.")
            return original_normalize_float(value, name=name)

        setattr(_normalize_float, _NORMALIZE_FLOAT_PATCH_MARKER, True)
        source_outlier._normalize_float = _normalize_float

    original_compute = source_outlier.compute_source_outlier_weights
    if getattr(original_compute, _PATCH_MARKER, False):
        return

    @wraps(original_compute)
    def compute_source_outlier_weights(
        source_features: Any,
        source_labels: Any,
        *,
        config: Any = None,
    ):
        cfg = source_outlier.source_outlier_config() if config is None else source_outlier._coerce_config(config)
        features = source_outlier._feature_matrix(source_features, name="source_features")
        labels = source_outlier._label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
        classes, _class_counts = label_counts(labels)
        if classes.shape[0] < 2:
            raise ValueError("At least two source classes are required.")

        class_labels = classes.tolist()
        class_masks = [label_equal_mask(labels, label) for label in class_labels]
        centroids = np.vstack([np.mean(features[mask], axis=0) for mask in class_masks])
        class_index_for_row = np.empty(labels.shape[0], dtype=int)
        assigned = np.zeros(labels.shape[0], dtype=bool)
        for class_index, mask in enumerate(class_masks):
            class_index_for_row[mask] = class_index
            assigned |= mask
        if not np.all(assigned):  # pragma: no cover - defensive guard for custom label objects
            raise ValueError("Every source row must match exactly one source class.")

        scale = source_outlier._feature_scale(features, enabled=cfg.use_diagonal_scale, epsilon=cfg.epsilon)
        centroid_rows = centroids[class_index_for_row]
        distances = source_outlier._scaled_l2(features, centroid_rows, scale=scale)
        threshold_values = np.asarray(
            [
                source_outlier._class_threshold(distances[mask], cfg=cfg)
                for mask in class_masks
            ],
            dtype=float,
        )
        threshold_rows = threshold_values[class_index_for_row]
        inlier_mask = distances <= threshold_rows
        weights = source_outlier._weights(distances, threshold_rows, cfg=cfg)
        thresholds = _LabelThresholdMapping(class_labels, threshold_values)

        return source_outlier.SourceOutlierResult(
            distances=distances.astype(np.float32, copy=False),
            sample_weights=weights.astype(np.float32, copy=False),
            inlier_mask=inlier_mask,
            thresholds=thresholds,
            classes=classes,
            centroids=centroids.astype(np.float32, copy=False),
            feature_scale=scale.astype(np.float32, copy=False),
            metadata=source_outlier._metadata(
                cfg,
                labels=labels,
                classes=classes,
                distances=distances,
                inlier_mask=inlier_mask,
                thresholds=thresholds,
            ),
        )

    setattr(compute_source_outlier_weights, _PATCH_MARKER, True)
    source_outlier._unique_label_vector = _unique_label_vector
    source_outlier._label_equal_mask = _label_equal_mask
    source_outlier.compute_source_outlier_weights = compute_source_outlier_weights


__all__ = ["install"]
