"""Handle NaN/composite labels consistently in source-outlier weighting."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask, values_equal

_PATCH_MARKER = "_neureptrace_source_outlier_nan_labels_patch_installed"


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


def _float32_if_safe(values: Any) -> np.ndarray:
    """Use float32 unless conversion overflows or erases nonzero values."""

    array = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    lost_finite = np.isfinite(array) & ~np.isfinite(compact)
    lost_nonzero = (array != 0.0) & (compact == 0.0)
    if bool(np.any(lost_finite | lost_nonzero)):
        return array
    return compact


def install() -> None:
    """Install NaN-aware source-label grouping for source-outlier weighting."""

    import neureptrace.decoding.source_outlier as source_outlier

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
            distances=_float32_if_safe(distances),
            sample_weights=_float32_if_safe(weights),
            inlier_mask=inlier_mask,
            thresholds=thresholds,
            classes=classes,
            centroids=_float32_if_safe(centroids),
            feature_scale=_float32_if_safe(scale),
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
