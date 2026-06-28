"""Result aggregation patches for metric validation and optional metrics."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

_PATCH_ATTR = "_neureptrace_rejects_bool_time_decode_metrics"
_OPTIONAL_PATCH_ATTR = "_neureptrace_results_optional_metric_aggregation"
_OPTIONAL_METRICS = ("balanced_accuracy", "top2_accuracy", "top3_accuracy")


def _bool_mask(values: pd.Series) -> pd.Series:
    return values.map(lambda value: isinstance(value, (bool, np.bool_))).fillna(False).astype(bool)


def _optional_metric_columns(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in _OPTIONAL_METRICS if metric in frame.columns and frame[metric].notna().any()]


def _metric_columns(results_module, frame: pd.DataFrame) -> list[str]:
    metrics = list(results_module.METRIC_COLUMNS)
    metrics.extend(metric for metric in _optional_metric_columns(frame) if metric not in metrics)
    return metrics


def _subject_time_keys(results_module, results_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normalized = results_module._normalize_emission_mode(results_frame)
    group_columns = [column for column in results_module.SUMMARY_GROUP_COLUMNS if column in normalized.columns]
    return normalized, [*group_columns, "subject", "time"]


def _optional_subject_time_metrics(
    results_module,
    results_frame: pd.DataFrame,
    *,
    optional_metrics: Sequence[str],
) -> pd.DataFrame | None:
    normalized, subject_time_keys = _subject_time_keys(results_module, results_frame)
    optional_frames: list[pd.DataFrame] = []
    for metric in optional_metrics:
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
        merged = merged.merge(optional_frame, on=subject_time_keys, how="outer", validate="one_to_one")
    return merged.sort_values(subject_time_keys).reset_index(drop=True)


def install() -> None:
    """Install aggregate result metric guards."""

    results = importlib.import_module("neureptrace.results")

    original = results._coerce_finite_metric_columns
    if not getattr(original, _PATCH_ATTR, False):

        def _coerce_finite_metric_columns_checked(
            frame: pd.DataFrame,
            metric_columns: Sequence[str],
        ) -> pd.DataFrame:
            for metric in metric_columns:
                if metric not in frame.columns:
                    continue
                bad_values = _bool_mask(frame[metric])
                if bad_values.any():
                    rows = bad_values[bad_values].index.tolist()[:5]
                    raise ValueError(f"Metric column '{metric}' must not contain booleans; bad row(s): {rows}.")
            return original(frame, metric_columns)

        setattr(_coerce_finite_metric_columns_checked, _PATCH_ATTR, True)
        _coerce_finite_metric_columns_checked.__wrapped__ = original
        results._coerce_finite_metric_columns = _coerce_finite_metric_columns_checked

    if getattr(results, _OPTIONAL_PATCH_ATTR, False):
        return

    original_subject_time_metrics = results.subject_time_metrics
    original_aggregate_time_decode_results = results.aggregate_time_decode_results

    def subject_time_metrics_with_optional_metrics(
        results_frame: pd.DataFrame,
        *,
        observations: pd.DataFrame | None = None,
        metric_columns: Sequence[str] | str | None = None,
        ece_bins: int = results.DEFAULT_ECE_BINS,
    ) -> pd.DataFrame:
        if metric_columns is not None:
            return original_subject_time_metrics(
                results_frame,
                observations=observations,
                metric_columns=metric_columns,
                ece_bins=ece_bins,
            )

        optional_metrics = _optional_metric_columns(results_frame)
        if not optional_metrics:
            return original_subject_time_metrics(results_frame, observations=observations, ece_bins=ece_bins)

        subject_time = original_subject_time_metrics(results_frame, observations=observations, ece_bins=ece_bins)
        optional_subject_time = _optional_subject_time_metrics(results, results_frame, optional_metrics=optional_metrics)
        if optional_subject_time is None:
            return subject_time

        group_columns = [column for column in results.SUMMARY_GROUP_COLUMNS if column in subject_time.columns]
        subject_time_keys = [*group_columns, "subject", "time"]
        return subject_time.merge(optional_subject_time, on=subject_time_keys, how="left", validate="one_to_one").sort_values(
            subject_time_keys
        ).reset_index(drop=True)

    def aggregate_time_decode_results_with_optional_metrics(
        results_frame: pd.DataFrame,
        *,
        observations: pd.DataFrame | None = None,
        ece_bins: int = results.DEFAULT_ECE_BINS,
    ) -> pd.DataFrame:
        optional_metrics = _optional_metric_columns(results_frame)
        if not optional_metrics:
            return original_aggregate_time_decode_results(results_frame, observations=observations, ece_bins=ece_bins)

        subject_time = subject_time_metrics_with_optional_metrics(results_frame, observations=observations, ece_bins=ece_bins)
        metric_columns = _metric_columns(results, subject_time)
        group_columns = [column for column in results.SUMMARY_GROUP_COLUMNS if column in subject_time.columns]
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
    results.subject_time_metrics = subject_time_metrics_with_optional_metrics
    results.aggregate_time_decode_results = aggregate_time_decode_results_with_optional_metrics
    setattr(results, _OPTIONAL_PATCH_ATTR, True)


__all__ = ["install"]
