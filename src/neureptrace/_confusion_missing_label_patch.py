"""Runtime patch for missing-label equality in confusion metrics."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_confusion_missing_label_patch_installed"
_MATCHING_MISSING = object()


def _is_missing_label_scalar(value: Any) -> bool:
    """Return whether *value* is a scalar NaN/NA/NaT label.

    ``None`` remains a distinct label because ordinary Python equality already
    handles ``None == None`` and existing label utilities treat it separately
    from pandas/NumPy missing scalars.
    """

    if value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, float) and bool(np.isnan(value))


def _labels_equal(left: Any, right: Any) -> bool:
    if _is_missing_label_scalar(left) and _is_missing_label_scalar(right):
        return True
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(equal, (bool, np.bool_)):
        return bool(equal)
    try:
        return bool(np.all(equal))
    except (TypeError, ValueError):
        return False


def _matching_missing_mask(left: Sequence[Any], right: Sequence[Any]) -> np.ndarray:
    return np.fromiter(
        (_is_missing_label_scalar(left_value) and _is_missing_label_scalar(right_value) for left_value, right_value in zip(left, right, strict=True)),
        dtype=bool,
        count=len(left),
    )


def _prepare_matching_missing_rows(frame: pd.DataFrame, *, true_column: str, predicted_column: str) -> pd.DataFrame:
    """Make only rows with two missing labels compare equal.

    One-sided missing labels remain untouched so genuine missing-to-class and
    class-to-missing errors retain their original label values in reports.
    """

    if true_column not in frame.columns or predicted_column not in frame.columns:
        return frame
    matching_missing = _matching_missing_mask(frame[true_column].tolist(), frame[predicted_column].tolist())
    if not np.any(matching_missing):
        return frame
    prepared = frame.copy()
    prepared[true_column] = prepared[true_column].astype(object)
    prepared[predicted_column] = prepared[predicted_column].astype(object)
    prepared.loc[matching_missing, true_column] = _MATCHING_MISSING
    prepared.loc[matching_missing, predicted_column] = _MATCHING_MISSING
    return prepared


def install() -> None:
    """Install missing-label-aware confusion metric wrappers."""

    confusion = importlib.import_module("neureptrace.metrics.confusion")
    if getattr(confusion, _PATCH_MARKER, False):
        return

    original_per_class_accuracy = confusion.per_class_accuracy
    original_confusion_pair_summary = confusion.confusion_pair_summary
    original_confusion_category_enrichment = confusion.confusion_category_enrichment
    original_confusion_category_matrix = confusion.confusion_category_matrix

    @wraps(original_per_class_accuracy)
    def per_class_accuracy(
        frame: pd.DataFrame,
        true_column: str = "true_label",
        predicted_column: str = "predicted_label",
        participant_column: str | None = None,
        group_columns: Sequence[str] = (),
    ) -> pd.DataFrame:
        group_columns = confusion._normalize_columns(group_columns)
        required_columns = [true_column, predicted_column, *group_columns]
        if participant_column is not None:
            required_columns.append(participant_column)
        confusion._require_columns(frame, required_columns)

        working_columns = [*group_columns, true_column, predicted_column]
        if participant_column is not None:
            working_columns.append(participant_column)
        working = frame[working_columns].rename(columns={true_column: "true_label", predicted_column: "predicted_label"})
        working["_correct"] = np.fromiter(
            (_labels_equal(true_label, predicted_label) for true_label, predicted_label in zip(working["true_label"], working["predicted_label"], strict=True)),
            dtype=bool,
            count=len(working),
        )

        rows: list[dict[str, object]] = []
        keys = [*group_columns, "true_label"]
        for group_key, group in working.groupby(keys, dropna=False, sort=True):
            row = confusion._group_row(keys, group_key)
            row.update(
                {
                    "n_trials": int(len(group)),
                    "n_correct": int(group["_correct"].sum()),
                    "accuracy": float(group["_correct"].mean()),
                }
            )
            if participant_column is not None:
                row["n_participants"] = int(group[participant_column].nunique(dropna=True))
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)

    @wraps(original_confusion_pair_summary)
    def confusion_pair_summary(
        frame: pd.DataFrame,
        true_column: str = "true_label",
        predicted_column: str = "predicted_label",
        group_columns: Sequence[str] = (),
        participant_column: str | None = None,
        metadata_frame: pd.DataFrame | None = None,
        metadata_label_columns: Sequence[str] = confusion.DEFAULT_METADATA_LABEL_COLUMNS,
        label_prefix: str = "label",
    ) -> pd.DataFrame:
        return original_confusion_pair_summary(
            _prepare_matching_missing_rows(frame, true_column=true_column, predicted_column=predicted_column),
            true_column=true_column,
            predicted_column=predicted_column,
            group_columns=group_columns,
            participant_column=participant_column,
            metadata_frame=metadata_frame,
            metadata_label_columns=metadata_label_columns,
            label_prefix=label_prefix,
        )

    @wraps(original_confusion_category_enrichment)
    def confusion_category_enrichment(
        frame: pd.DataFrame,
        *,
        metadata_frame: pd.DataFrame,
        true_column: str = "true_label",
        predicted_column: str = "predicted_label",
        category_columns: Sequence[str] | str | None = None,
        group_columns: Sequence[str] = (),
        participant_column: str | None = None,
        metadata_label_columns: Sequence[str] = confusion.DEFAULT_METADATA_LABEL_COLUMNS,
        n_permutations: int | None = 10_000,
        seed: int | None = 0,
    ) -> pd.DataFrame:
        return original_confusion_category_enrichment(
            _prepare_matching_missing_rows(frame, true_column=true_column, predicted_column=predicted_column),
            metadata_frame=metadata_frame,
            true_column=true_column,
            predicted_column=predicted_column,
            category_columns=category_columns,
            group_columns=group_columns,
            participant_column=participant_column,
            metadata_label_columns=metadata_label_columns,
            n_permutations=n_permutations,
            seed=seed,
        )

    @wraps(original_confusion_category_matrix)
    def confusion_category_matrix(
        frame: pd.DataFrame,
        *,
        metadata_frame: pd.DataFrame,
        true_column: str = "true_label",
        predicted_column: str = "predicted_label",
        category_columns: Sequence[str] | str | None = None,
        group_columns: Sequence[str] = (),
        participant_column: str | None = None,
        metadata_label_columns: Sequence[str] = confusion.DEFAULT_METADATA_LABEL_COLUMNS,
    ) -> pd.DataFrame:
        return original_confusion_category_matrix(
            _prepare_matching_missing_rows(frame, true_column=true_column, predicted_column=predicted_column),
            metadata_frame=metadata_frame,
            true_column=true_column,
            predicted_column=predicted_column,
            category_columns=category_columns,
            group_columns=group_columns,
            participant_column=participant_column,
            metadata_label_columns=metadata_label_columns,
        )

    confusion.per_class_accuracy = per_class_accuracy
    confusion.confusion_pair_summary = confusion_pair_summary
    confusion.confusion_category_enrichment = confusion_category_enrichment
    confusion.confusion_category_matrix = confusion_category_matrix
    setattr(confusion, _PATCH_MARKER, True)

    metrics_module = sys.modules.get("neureptrace.metrics")
    if metrics_module is not None:
        metrics_module.per_class_accuracy = per_class_accuracy
        metrics_module.confusion_pair_summary = confusion_pair_summary
        metrics_module.confusion_category_enrichment = confusion_category_enrichment
        metrics_module.confusion_category_matrix = confusion_category_matrix


__all__ = ["install"]
