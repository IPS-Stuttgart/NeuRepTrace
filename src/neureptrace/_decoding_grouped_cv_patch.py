"""Runtime guardrail for grouped cross-validation feasibility.

Grouped decoding folds fail late, or silently tune on degenerate folds, when a
class appears in only one group.  In that case the fold that holds out that
group leaves the training split without that class.  This patch keeps grouped
outer CV and grouped inner tuning from constructing such splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_grouped_cv_patch_installed"


def _as_1d_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _minimum_class_group_count(labels: Any, groups: Any) -> int:
    labels_array = _as_1d_array(labels, name="labels")
    groups_array = _as_1d_array(groups, name="groups")
    if labels_array.shape[0] != groups_array.shape[0]:
        raise ValueError("labels and groups must contain the same number of rows.")
    class_group_counts = [int(np.unique(groups_array[labels_array == label]).shape[0]) for label in np.unique(labels_array)]
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
        if groups is not None:
            _validate_grouped_class_coverage(labels, groups)
        return original_make_cross_validator(labels, groups, n_splits)

    def make_tuning_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
        labels_array = _as_1d_array(labels, name="labels")
        groups_array = None if groups is None else _as_1d_array(groups, name="groups")
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
