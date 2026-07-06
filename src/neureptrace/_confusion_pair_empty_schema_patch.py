"""Runtime patch for stable empty confusion-pair summary schemas."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from functools import wraps

import pandas as pd

from neureptrace.metrics.confusion import DEFAULT_METADATA_LABEL_COLUMNS

_PATCH_MARKER = "_neureptrace_confusion_pair_empty_schema_patched"

_BASE_CONFUSION_PAIR_COLUMNS = (
    "a_to_b_count",
    "b_to_a_count",
    "total_confusions",
    "true_a_trials",
    "true_b_trials",
    "a_to_b_rate",
    "b_to_a_rate",
    "mean_directional_rate",
    "max_directional_rate",
    "min_directional_rate",
    "absolute_rate_asymmetry",
    "total_pair_error_rate",
    "true_a_error_count",
    "true_b_error_count",
    "predicted_a_error_count",
    "predicted_b_error_count",
    "expected_a_to_b_count",
    "expected_b_to_a_count",
    "expected_total_confusions",
    "a_to_b_lift",
    "b_to_a_lift",
    "pair_confusion_lift",
    "total_confusion_excess",
    "pair_standardized_residual",
    "symmetric_confusion_count",
    "n_confused_participants",
    "a_to_b_participants",
    "b_to_a_participants",
)


def _normalize_columns(columns: Sequence[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(dict.fromkeys(str(column) for column in columns))


def _metadata_schema_columns(
    metadata_frame: pd.DataFrame | None,
    metadata_label_columns: Sequence[str],
    label_prefix: str,
) -> list[str]:
    if metadata_frame is None or metadata_frame.empty:
        return []

    excluded = set(_normalize_columns(metadata_label_columns))
    columns: list[str] = []
    for key in sorted(str(column) for column in metadata_frame.columns if str(column) not in excluded):
        columns.extend((f"{label_prefix}_a_{key}", f"{label_prefix}_b_{key}", f"same_{key}"))
    return columns


def _empty_confusion_pair_summary_frame(
    *,
    group_columns: Sequence[str] | str | None,
    metadata_frame: pd.DataFrame | None,
    metadata_label_columns: Sequence[str],
    label_prefix: str,
) -> pd.DataFrame:
    columns = [
        *_normalize_columns(group_columns),
        f"{label_prefix}_a",
        f"{label_prefix}_b",
        *_BASE_CONFUSION_PAIR_COLUMNS,
        *_metadata_schema_columns(metadata_frame, metadata_label_columns, label_prefix),
    ]
    return pd.DataFrame(columns=columns)


def install() -> None:
    """Install stable schemas for no-error confusion-pair summaries."""
    importlib.import_module("neureptrace._ranking_score_iterables_patch").install()
    import neureptrace.metrics.confusion as confusion

    if getattr(confusion.confusion_pair_summary, _PATCH_MARKER, False):
        return

    original_confusion_pair_summary = confusion.confusion_pair_summary

    @wraps(original_confusion_pair_summary)
    def confusion_pair_summary(
        frame: pd.DataFrame,
        true_column: str = "true_label",
        predicted_column: str = "predicted_label",
        group_columns: Sequence[str] = (),
        participant_column: str | None = None,
        metadata_frame: pd.DataFrame | None = None,
        metadata_label_columns: Sequence[str] = DEFAULT_METADATA_LABEL_COLUMNS,
        label_prefix: str = "label",
    ) -> pd.DataFrame:
        summary = original_confusion_pair_summary(
            frame,
            true_column=true_column,
            predicted_column=predicted_column,
            group_columns=group_columns,
            participant_column=participant_column,
            metadata_frame=metadata_frame,
            metadata_label_columns=metadata_label_columns,
            label_prefix=label_prefix,
        )
        if summary.empty and len(summary.columns) == 0:
            return _empty_confusion_pair_summary_frame(
                group_columns=group_columns,
                metadata_frame=metadata_frame,
                metadata_label_columns=metadata_label_columns,
                label_prefix=label_prefix,
            )
        return summary

    setattr(confusion_pair_summary, _PATCH_MARKER, True)
    confusion.confusion_pair_summary = confusion_pair_summary

    metrics_module = sys.modules.get("neureptrace.metrics")
    if metrics_module is not None:
        metrics_module.confusion_pair_summary = confusion_pair_summary


__all__ = ["install"]
