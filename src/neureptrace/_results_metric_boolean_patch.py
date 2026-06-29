"""Result aggregation patches for metric validation and optional metrics."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

_PATCH_ATTR = "_neureptrace_rejects_bool_time_decode_metrics"
_OBSERVATION_PATCH_ATTR = "_neureptrace_rejects_bool_probability_observations"
_SUMMARY_PATCH_ATTR = "_neureptrace_rejects_bool_metric_table_inputs"
_OPTIONAL_PATCH_ATTR = "_neureptrace_results_optional_metric_aggregation"
_OPTIONAL_METRICS = ("balanced_accuracy", "top2_accuracy", "top3_accuracy")


def _bool_mask(values: pd.Series) -> pd.Series:
    return values.map(lambda value: isinstance(value, (bool, np.bool_))).fillna(False).astype(bool)


def _boolean_rows(values: pd.Series) -> list[object]:
    mask = _bool_mask(values)
    return mask[mask].index.tolist()[:5]


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


def _normalize_optional_columns(columns: Sequence[str] | str | None) -> tuple[str, ...]:
    if columns is None:
        return ()
    if isinstance(columns, str):
        return (columns,)
    return tuple(dict.fromkeys(columns))


def _reject_boolean_table_column(frame: pd.DataFrame, column: str | None) -> None:
    if column is None or column not in frame.columns:
        return
    rows = _boolean_rows(frame[column])
    if rows:
        raise ValueError(f"Metric table column '{column}' must not contain booleans; bad row(s): {rows}.")


def _group_column_names(group_columns: Sequence[str] | str | None) -> tuple[str, ...]:
    if group_columns is None:
        return ()
    if isinstance(group_columns, str):
        return (group_columns,)
    return tuple(group_columns)


def _reject_mixed_boolean_optional_column(
    frame: pd.DataFrame,
    column: str | None,
    group_columns: Sequence[str] | str | None,
) -> None:
    if column is None or column not in frame.columns:
        return
    rows = _boolean_rows(frame[column])
    if not rows:
        return

    groups = _group_column_names(group_columns)
    grouped_frames = [((), frame)] if not groups else frame.groupby(list(groups), dropna=False, sort=False)
    for _, group in grouped_frames:
        values = group[column].dropna()
        if values.empty or not _bool_mask(values).any():
            continue
        if (~_bool_mask(values)).any():
            bad_rows = _boolean_rows(group[column])
            raise ValueError(f"Metric table column '{column}' must not contain booleans; bad row(s): {bad_rows}.")


def install() -> None:
    """Install aggregate result metric guards."""

    results = importlib.import_module("neureptrace.results")
    tables = importlib.import_module("neureptrace.results.tables")

    original = results._coerce_finite_metric_columns
    if not getattr(original, _PATCH_ATTR, False):

        def _coerce_finite_metric_columns_checked(
            frame: pd.DataFrame,
            metric_columns: Sequence[str],
        ) -> pd.DataFrame:
            for metric in metric_columns:
                if metric not in frame.columns:
                    continue
                rows = _boolean_rows(frame[metric])
                if rows:
                    raise ValueError(f"Metric column '{metric}' must not contain booleans; bad row(s): {rows}.")
            return original(frame, metric_columns)

        setattr(_coerce_finite_metric_columns_checked, _PATCH_ATTR, True)
        _coerce_finite_metric_columns_checked.__wrapped__ = original
        results._coerce_finite_metric_columns = _coerce_finite_metric_columns_checked

    original_probability_ece_by_group = results._probability_ece_by_group
    if not getattr(original_probability_ece_by_group, _OBSERVATION_PATCH_ATTR, False):

        def _probability_ece_by_group_checked(
            observations: pd.DataFrame,
            group_columns: list[str],
            *,
            n_bins: int,
        ) -> pd.DataFrame:
            probability_columns = tuple(results.probability_columns(observations))
            for column in ("time", "true_label", *probability_columns):
                if column not in observations.columns:
                    continue
                rows = _boolean_rows(observations[column])
                if rows:
                    raise ValueError(f"Probability-observation column '{column}' must not contain booleans; bad row(s): {rows}.")
            return original_probability_ece_by_group(observations, group_columns, n_bins=n_bins)

        setattr(_probability_ece_by_group_checked, _OBSERVATION_PATCH_ATTR, True)
        _probability_ece_by_group_checked.__wrapped__ = original_probability_ece_by_group
        results._probability_ece_by_group = _probability_ece_by_group_checked

    original_summarize_metric_table = tables.summarize_metric_table
    if not getattr(original_summarize_metric_table, _SUMMARY_PATCH_ATTR, False):

        def summarize_metric_table_checked(
            frame: pd.DataFrame,
            value_column: str,
            group_columns: Sequence[str] | str | None,
            participant_column: str | None = None,
            chance_column: str | None = None,
            scale: float = 1.0,
            *,
            percent_scale: float | None = None,
            percent_prefix: str = "percent",
            chance_percent_column: str | None = None,
            chance_class_columns: Sequence[str] | str | None = None,
            permutation_p_column: str | None = None,
            p_value_thresholds: Sequence[float] = (0.05, 0.01),
            zero_singleton_dispersion: bool = False,
        ) -> pd.DataFrame:
            for column in (value_column, permutation_p_column):
                _reject_boolean_table_column(frame, column)
            for column in (chance_column, *_normalize_optional_columns(chance_class_columns)):
                _reject_mixed_boolean_optional_column(frame, column, group_columns)
            return original_summarize_metric_table(
                frame,
                value_column,
                group_columns,
                participant_column=participant_column,
                chance_column=chance_column,
                scale=scale,
                percent_scale=percent_scale,
                percent_prefix=percent_prefix,
                chance_percent_column=chance_percent_column,
                chance_class_columns=chance_class_columns,
                permutation_p_column=permutation_p_column,
                p_value_thresholds=p_value_thresholds,
                zero_singleton_dispersion=zero_singleton_dispersion,
            )

        setattr(summarize_metric_table_checked, _SUMMARY_PATCH_ATTR, True)
        summarize_metric_table_checked.__wrapped__ = original_summarize_metric_table
        tables.summarize_metric_table = summarize_metric_table_checked
        results.summarize_metric_table = summarize_metric_table_checked
    else:
        results.summarize_metric_table = original_summarize_metric_table

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
