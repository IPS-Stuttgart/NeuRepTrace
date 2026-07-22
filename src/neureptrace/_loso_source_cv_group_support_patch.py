"""Guard LOSO group handling and grouped source-window CV support."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

_CV_PATCH_MARKER = "_neureptrace_loso_source_cv_group_support_patch_installed"
_GROUP_PATCH_MARKER = "_neureptrace_loso_group_label_validation_patch_installed"


def _minimum_class_group_support(labels: np.ndarray, groups: np.ndarray) -> int:
    """Return the fewest distinct source groups available for any class."""

    supports = [len(np.unique(groups[labels == label])) for label in np.unique(labels)]
    return int(min(supports)) if supports else 0


def _all_folds_cover_classes(splits: list[tuple[np.ndarray, np.ndarray]], labels: np.ndarray, n_classes: int) -> bool:
    """Check that every generated train and validation fold contains all classes."""

    return all(
        len(np.unique(labels[train_indices])) == n_classes and len(np.unique(labels[val_indices])) == n_classes
        for train_indices, val_indices in splits
    )


def _reject_missing_groups(groups: Any, *, name: str) -> None:
    """Reject group labels that cannot define a valid held-out partition."""

    missing = np.asarray(pd.isna(groups), dtype=bool)
    if missing.any():
        raise ValueError(f"{name} must not contain missing values.")


def _install_source_cv_patch(module: Any) -> None:
    original = module._feasible_source_cv_splits
    if getattr(original, _CV_PATCH_MARKER, False):
        return

    def _feasible_source_cv_splits(labels: np.ndarray, groups: np.ndarray | None, requested_splits: int):
        labels = np.asarray(labels).reshape(-1)
        _, class_counts = np.unique(labels, return_counts=True)
        n_classes = len(class_counts)
        if n_classes < 2:
            raise ValueError("Need at least two classes for source-window selection.")

        feasible_splits = min(int(requested_splits), int(np.min(class_counts)))
        if groups is None:
            if feasible_splits < 2:
                raise ValueError("Need at least two inner folds for source-window selection.")
            return StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=13).split(
                np.zeros((len(labels), 1)),
                labels,
            )

        groups = np.asarray(groups).reshape(-1)
        if len(groups) != len(labels):
            raise ValueError("Source-window selection labels and groups must have the same length.")
        _reject_missing_groups(groups, name="Source-window selection groups")

        class_group_support = _minimum_class_group_support(labels, groups)
        if class_group_support < 2:
            raise ValueError(
                "Grouped source-window selection requires every class to appear in at least two source groups. "
                "At least one class is isolated to a single source group, so an inner training fold would be single-class."
            )

        feasible_splits = min(feasible_splits, len(np.unique(groups)), class_group_support)
        if feasible_splits < 2:
            raise ValueError("Need at least two inner folds for source-window selection.")

        for n_splits in range(feasible_splits, 1, -1):
            splitter = StratifiedGroupKFold(n_splits=n_splits)
            splits = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
            if _all_folds_cover_classes(splits, labels, n_classes):
                return iter(splits)

        raise ValueError("Could not build grouped source-window selection folds where every train and validation fold contains all classes.")

    setattr(_feasible_source_cv_splits, _CV_PATCH_MARKER, True)
    module._feasible_source_cv_splits = _feasible_source_cv_splits


def _install_group_label_patch(module: Any) -> None:
    original = module._preprocessed_data_for_outer_fold
    if getattr(original, _GROUP_PATCH_MARKER, False):
        return

    @wraps(original)
    def _preprocessed_data_for_outer_fold(
        data: np.ndarray,
        times: np.ndarray,
        metadata: pd.DataFrame,
        *,
        normalization: str,
        normalization_scope: str,
        baseline_window: tuple[float, float],
        train_indices: np.ndarray,
        group_column: str,
    ) -> np.ndarray:
        _reject_missing_groups(metadata[group_column], name=f"LOSO group column '{group_column}'")
        return original(
            data,
            times,
            metadata,
            normalization=normalization,
            normalization_scope=normalization_scope,
            baseline_window=baseline_window,
            train_indices=train_indices,
            group_column=group_column,
        )

    setattr(_preprocessed_data_for_outer_fold, _GROUP_PATCH_MARKER, True)
    module._preprocessed_data_for_outer_fold = _preprocessed_data_for_outer_fold


def install() -> None:
    """Reject missing LOSO groups and invalid grouped source-window folds."""

    module = importlib.import_module("neureptrace.loso_time_decode")
    _install_source_cv_patch(module)
    _install_group_label_patch(module)


__all__ = ["install"]
