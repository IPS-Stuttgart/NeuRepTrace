"""Runtime guardrails for grouped cross-validation feasibility.

Grouped decoding folds fail late, or silently tune on degenerate folds, when a
class appears in only one group. In that case the fold that holds out that
group leaves the training split without that class. Composite class labels or
group identifiers also need to be represented as one atomic value per row;
otherwise NumPy expands tuples and row arrays into a two-dimensional target that
scikit-learn interprets as multi-output classification.

This patch validates grouped outer CV and grouped inner tuning before splits are
constructed and dense-encodes composite identifiers only for split generation.
The returned train/test indices retain their original row semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neureptrace._object_label_utils import values_equal

_PATCH_MARKER = "_neureptrace_grouped_cv_patch_installed"


def _top_level_items(values: Any, *, name: str) -> list[object]:
    """Materialize one logical label or group identifier per input row."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain one value per row, not a string scalar.")
    if isinstance(values, np.ndarray):
        array = np.asarray(values)
        if array.ndim == 0:
            return [array[()]]
        return [array[index] for index in range(array.shape[0])]
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain one value per row.") from exc


def _is_composite(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (tuple, list, dict))


def _stable_encode(items: list[object]) -> np.ndarray:
    """Dense-encode arbitrary row identifiers using semantic equality."""

    classes: list[object] = []
    encoded = np.empty(len(items), dtype=np.int64)
    for row_index, item in enumerate(items):
        for class_index, class_item in enumerate(classes):
            if values_equal(item, class_item):
                encoded[row_index] = class_index
                break
        else:
            encoded[row_index] = len(classes)
            classes.append(item)
    return encoded


def _cv_vector(values: Any, *, name: str) -> np.ndarray:
    """Return a one-dimensional scalar vector suitable for sklearn splitters."""

    items = _top_level_items(values, name=name)
    if not items:
        return np.asarray([], dtype=np.int64)

    if not any(_is_composite(item) for item in items):
        array = np.asarray(items)
        if array.ndim == 1 and array.dtype != object:
            return array

    return _stable_encode(items)


def _minimum_class_group_count(labels: Any, groups: Any) -> int:
    labels_array = _cv_vector(labels, name="labels")
    groups_array = _cv_vector(groups, name="groups")
    if labels_array.shape[0] != groups_array.shape[0]:
        raise ValueError("labels and groups must contain the same number of rows.")
    class_group_counts = [
        int(np.unique(groups_array[labels_array == label]).shape[0])
        for label in np.unique(labels_array)
    ]
    return min(class_group_counts, default=0)


def _validate_grouped_class_coverage(labels: Any, groups: Any) -> None:
    min_class_groups = _minimum_class_group_count(labels, groups)
    if min_class_groups < 2:
        raise ValueError(
            "Need each class to appear in at least two groups for grouped CV; "
            f"the sparsest class appears in {min_class_groups} group(s)."
        )


def install() -> None:
    """Install grouped-CV guards for decoder split helpers."""

    from neureptrace import decoding

    if getattr(decoding, _PATCH_MARKER, False):
        return

    original_make_cross_validator = decoding.make_cross_validator

    def make_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
        labels_array = _cv_vector(labels, name="labels")
        groups_array = None if groups is None else _cv_vector(groups, name="groups")
        if groups_array is not None:
            if labels_array.shape[0] != groups_array.shape[0]:
                raise ValueError("labels and groups must contain the same number of rows.")
            _validate_grouped_class_coverage(labels_array, groups_array)
        return original_make_cross_validator(labels_array, groups_array, n_splits)

    def make_tuning_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
        labels_array = _cv_vector(labels, name="labels")
        groups_array = None if groups is None else _cv_vector(groups, name="groups")
        _, class_counts = np.unique(labels_array, return_counts=True)
        if len(class_counts) < 2:
            raise ValueError("Need at least two classes for decoder hyperparameter tuning.")

        feasible_splits = min(int(n_splits), int(np.min(class_counts)))
        if groups_array is not None:
            if labels_array.shape[0] != groups_array.shape[0]:
                raise ValueError("labels and groups must contain the same number of rows.")
            min_class_groups = _minimum_class_group_count(labels_array, groups_array)
            if min_class_groups < 2:
                raise ValueError(
                    "Need each class to appear in at least two groups when grouped to tune decoder hyperparameters; "
                    f"the sparsest class appears in {min_class_groups} group(s)."
                )
            feasible_splits = min(feasible_splits, len(np.unique(groups_array)), min_class_groups)

        if feasible_splits < 2:
            raise ValueError("Need at least two examples per class to tune decoder hyperparameters.")
        return list(make_cross_validator(labels_array, groups_array, feasible_splits))

    make_cross_validator.__doc__ = original_make_cross_validator.__doc__
    make_tuning_cross_validator.__doc__ = decoding.make_tuning_cross_validator.__doc__
    decoding.make_cross_validator = make_cross_validator
    decoding.make_tuning_cross_validator = make_tuning_cross_validator
    setattr(decoding, _PATCH_MARKER, True)
