"""Result aggregation patches for metric validation and optional metrics."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

_BOOLEAN_PATCH_ATTR = "_neureptrace_rejects_boolean_time_decode_metrics"
_OPTIONAL_PATCH_ATTR = "_neureptrace_results_optional_metric_aggregation"
_OPTIONAL_METRIC_COLUMNS = ("balanced_accuracy", "top2_accuracy", "top3_accuracy")


def _boolean_mask(values: pd.Series) -> pd.Series:
    return values.map(lambda value: isinstance(value, (bool, np.bool_))).fillna(False).astype(bool)


def _optional_metric_columns_for(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in _OPTIONAL_METRIC_COLUMNS if metric in frame.columns and frame[metric].notna().any()]


def _metric_columns_for(results_module, frame: pd.DataFrame) -> list[str]:
    metric_columns = list(results_module.METRIC_COLUMNS)
    for metric in _optional_metric_columns_for(frame):
        if metric not in metric_columns:
            metric_columns.append(metric)
    return metric_columns


def _install_boolean_metric_guard(results_module) -> None:
    original = results_module._coerce_finite_metric_columns
    if getattr(original, _BOOLEAN_PATCH_ATTR, False):
        return

    def _coerce_finite_metric_columns_checked(
        frame: pd.DataFrame,
        metric_columns: Sequence[str],
    ) -> pd.DataFrame:
        for metric in metric_columns:
            if metric not in frame.columns:
                continue
            boolean_values = _boolean_mask(frame[metric])
            if boolean_values.any():
                bad_rows = boolean_values[boolean_values].index.tolist()[:5]
                raise ValueError(
                    f"Metric column '{metric}' must contain finite numeric values, not booleans; "
                    f"boolean row(s): {bad_rows}."
                )
        return original(frame, metric_columns)

    setattr(_coerce_finite_metric_columns_checked, _BOOLEAN_PATCH_ATTR, True)
    _coerce_finite_metric_columns_checked.__wrapped__ = original
    results_module._coerce_finite_metric_columns = _coerce_finite_metric_columns_checked


def _subject_time_keys(results_module, results_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normalized = results_module._normalize_emission_mode(results_frame)
    group_columns = [column for column in results_module.SUMMARY_GROUP_COLUMNS if column in normalized.columns]
    return normalized, [*group_columns, "subject", "time"]


def _optional_subject_time_metrics(
    results_module,
    results_frame: pd.DataFrame,
    *,
    optional_metric_columns: Sequence[str],
) -> pd.DataFrame | None:
    normalized, subject_time_keys = _subject_time_keys(results_module, results_frame)
    optional_frames: list[pd.DataFrame] = []
    for metric in optional_metric_columns:
        rows_with_metric = normalized.loc[normalized[metric].notna()].copy()
        if rows_with_metric.empty:
            continue
        optional_frames.append(
            results_module._mean_across_folds(
                rows_with_metric,
                subject_time_keys,
                metric_columns=[metric],
            )
        )

    if not optional_frames:
        return None

    merged = optional_frames[0]
    for optional_frame in optional_frames[1:]:
        merged = merged.merge(
            optional_frame,
            on=subject_time_keys,
            how="outer",
            validate="one_to_one",
        )
    return merged.sort_values(subject_time_keys).reset_index(drop=True)


def _install_optional_metric_aggregation(results_module) -> None:
    if getattr(results_module, _OPTIONAL_PATCH_ATTR, False):
        return

    original_subject_time_metrics = results_module.subject_time_metrics
    original_aggregate_time_decode_results = results_module.aggregate_time_decode_results

    def subject_time_metrics_with_optional_metrics(
        results_frame: pd.DataFrame,
        *,
        observations: pd.DataFrame | None = None,
        metric_columns: Sequence[str] | str | None = None,
        ece_bins: int = results_module.DEFAULT_ECE_BINS,
    ) -> pd.DataFrame:
        if metric_columns is not None:
            return original_subject_time_metrics(
                results_frame,
                observations=observations,
                metric_columns=metric_columns,
                ece_bins=ece_bins,
            )

        optional_metric_columns = _optional_metric_columns_for(results_frame)
        if not optional_metric_columns:
            return original_subject_time_metrics(
                results_frame,
                observations=observations,
                ece_bins=ece_bins,
            )

        subject_time = original_subject_time_metrics(
            results_frame,
            observations=observations,
            ece_bins=ece_bins,
        )
        optional_subject_time = _optional_subject_time_metrics(
            results_module,
            results_frame,
            optional_metric_columns=optional_metric_columns,
        )
        if optional_subject_time is None:
            return subject_time

        group_columns = [column for column in results_module.SUMMARY_GROUP_COLUMNS if column in subject_time.columns]
        subject_time_keys = [*group_columns, "subject", "time"]
        return subject_time.merge(
            optional_subject_time,
            on=subject_time_keys,
            how="left",
            validate="one_to_one",
        ).sort_values(subject_time_keys).reset_index(drop=True)

    def aggregate_time_decode_results_with_optional_metrics(
        results_frame: pd.DataFrame,
        *,
        observations: pd.DataFrame | None = None,
        ece_bins: int = results_module.DEFAULT_ECE_BINS,
    ) -> pd.DataFrame:
        optional_metric_columns = _optional_metric_columns_for(results_frame)
        if not optional_metric_columns:
            return original_aggregate_time_decode_results(
                results_frame,
                observations=observations,
                ece_bins=ece_bins,
            )

        subject_time = subject_time_metrics_with_optional_metrics(
            results_frame,
            observations=observations,
            ece_bins=ece_bins,
        )
        metric_columns = _metric_columns_for(results_module, subject_time)
        group_columns = [column for column in results_module.SUMMARY_GROUP_COLUMNS if column in subject_time.columns]
        aggregate_keys = [*group_columns, "time"]
        grouped = subject_time.groupby(aggregate_keys, as_index=False, dropna=False)
        aggregated = grouped[metric_columns].mean()
        n_subjects = grouped["subject"].nunique().rename(columns={"subject": "n_subjects"})
        aggregated = aggregated.merge(n_subjects, on=aggregate_keys)

        for metric in metric_columns:
            sem = grouped[metric].sem().rename(columns={metric: f"{metric}_sem"})
            aggregated = aggregated.merge(sem, on=aggregate_keys)
            aggregated = aggregated.rename(columns={metric: f"{metric}_mean"})

        return aggregated.sort_values(aggregate_keys).reset_index(drop=True)

    setattr(subject_time_metrics_with_optional_metrics, _OPTIONAL_PATCH_ATTR, True)
    setattr(aggregate_time_decode_results_with_optional_metrics, _OPTIONAL_PATCH_ATTR, True)
    subject_time_metrics_with_optional_metrics.__wrapped__ = original_subject_time_metrics
    aggregate_time_decode_results_with_optional_metrics.__wrapped__ = original_aggregate_time_decode_results
    results_module.subject_time_metrics = subject_time_metrics_with_optional_metrics
    results_module.aggregate_time_decode_results = aggregate_time_decode_results_with_optional_metrics
    setattr(results_module, _OPTIONAL_PATCH_ATTR, True)


def install() -> None:
    """Install result aggregation guards and optional metric preservation."""

    results_module = importlib.import_module("neureptrace.results")
    _install_boolean_metric_guard(results_module)
    _install_optional_metric_aggregation(results_module)


__all__ = ["install"]