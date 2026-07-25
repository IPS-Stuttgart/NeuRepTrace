"""Runtime patches for finite and positional metric selection."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps

import numpy as np
import pandas as pd

_REPORT_PATCH_MARKER = "_report_finite_metric_selection_patched"
_REPORT_AGGREGATE_PATCH_MARKER = "_report_aggregate_positional_selection_patched"
_RESULTS_METRIC_SELECTION_PATCH_MARKER = "_results_unique_metric_selection_patched"
_RESULTS_SUMMARY_SELECTION_PATCH_MARKER = "_results_summary_positional_selection_patched"
_SEMANTIC_STAGE_PATCH_MARKER = "_semantic_stage_positional_selection_patched"


def _finite_numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    return values.where(finite)


def _window_mean(frame: pd.DataFrame, column: str, start: float, stop: float) -> float:
    window = frame[(frame["time"] >= start) & (frame["time"] <= stop)]
    if window.empty:
        raise ValueError(f"No time points found in window [{start}, {stop}].")

    values = _finite_numeric_values(window, column).dropna()
    if values.empty:
        raise ValueError(f"No finite values found in column '{column}' for window [{start}, {stop}].")
    return float(values.mean())


def _best_metric_row(frame: pd.DataFrame, selection_metric: str, column: str) -> pd.Series:
    import neureptrace.report as report

    selection_metric = report._validate_selection_metric(selection_metric)
    if column not in frame.columns:
        raise ValueError(f"Frame is missing selection metric column '{column}'.")

    values = _finite_numeric_values(frame, column)
    finite_positions = np.flatnonzero(values.notna().to_numpy())
    if finite_positions.size == 0:
        raise ValueError(f"Selection metric column '{column}' contains no finite values.")

    finite_values = values.iloc[finite_positions].to_numpy(dtype=float)
    if report.METRIC_HIGHER_IS_BETTER[selection_metric]:
        selected_offset = int(np.argmax(finite_values))
    else:
        selected_offset = int(np.argmin(finite_values))
    return frame.iloc[int(finite_positions[selected_offset])]


def _install_report_patch() -> None:
    import neureptrace.report as report

    if not getattr(report._best_metric_row, _REPORT_PATCH_MARKER, False):
        setattr(_best_metric_row, _REPORT_PATCH_MARKER, True)
        setattr(_window_mean, _REPORT_PATCH_MARKER, True)
        report._best_metric_row = _best_metric_row
        report._window_mean = _window_mean

    original_summarize_aggregate_time_decode = report.summarize_aggregate_time_decode
    if getattr(original_summarize_aggregate_time_decode, _REPORT_AGGREGATE_PATCH_MARKER, False):
        return

    @wraps(original_summarize_aggregate_time_decode)
    def summarize_aggregate_time_decode(
        summary: pd.DataFrame,
        *,
        chance: float = 0.5,
        baseline_window: tuple[float, float] = (-0.1, 0.0),
        effect_window: tuple[float, float] = (0.1, 0.8),
        selection_metric: str = "accuracy",
    ) -> dict[str, float | str | bool]:
        if "accuracy_mean" in summary.columns:
            accuracy_values = _finite_numeric_values(summary, "accuracy_mean")
            if not bool(accuracy_values.notna().any()):
                raise ValueError("Accuracy mean column 'accuracy_mean' contains no finite values.")
            summary = summary.copy()
            summary["accuracy_mean"] = accuracy_values
        if not summary.index.is_unique:
            summary = summary.reset_index(drop=True)
        return original_summarize_aggregate_time_decode(
            summary,
            chance=chance,
            baseline_window=baseline_window,
            effect_window=effect_window,
            selection_metric=selection_metric,
        )

    setattr(summarize_aggregate_time_decode, _REPORT_AGGREGATE_PATCH_MARKER, True)
    report.summarize_aggregate_time_decode = summarize_aggregate_time_decode


def _install_semantic_stage_patch() -> None:
    import neureptrace.semantic_stages as semantic_stages

    original_detect_stable_stages = semantic_stages.detect_stable_stages
    if getattr(original_detect_stable_stages, _SEMANTIC_STAGE_PATCH_MARKER, False):
        return
    original_build_stage_report = semantic_stages.build_stage_report

    @wraps(original_detect_stable_stages)
    def detect_stable_stages(
        time_summary: pd.DataFrame,
        *,
        posterior_threshold: float = 0.6,
        match_threshold: float = 0.6,
        min_duration: float = 0.04,
    ) -> pd.DataFrame:
        return original_detect_stable_stages(
            time_summary.reset_index(drop=True),
            posterior_threshold=posterior_threshold,
            match_threshold=match_threshold,
            min_duration=min_duration,
        )

    @wraps(original_build_stage_report)
    def build_stage_report(
        time_summary: pd.DataFrame,
        stages: pd.DataFrame,
        *,
        posterior_threshold: float,
        match_threshold: float,
        min_duration: float,
    ) -> str:
        return original_build_stage_report(
            time_summary.reset_index(drop=True),
            stages,
            posterior_threshold=posterior_threshold,
            match_threshold=match_threshold,
            min_duration=min_duration,
        )

    setattr(detect_stable_stages, _SEMANTIC_STAGE_PATCH_MARKER, True)
    setattr(build_stage_report, _SEMANTIC_STAGE_PATCH_MARKER, True)
    semantic_stages.detect_stable_stages = detect_stable_stages
    semantic_stages.build_stage_report = build_stage_report


def _install_results_summary_selection_patch() -> None:
    import neureptrace.results as results

    original_best_summary_row = results._best_summary_row
    if getattr(original_best_summary_row, _RESULTS_SUMMARY_SELECTION_PATCH_MARKER, False):
        return

    @wraps(original_best_summary_row)
    def _best_summary_row(frame: pd.DataFrame, selection_metric: str) -> pd.Series:
        return original_best_summary_row(frame.reset_index(drop=True), selection_metric)

    setattr(_best_summary_row, _RESULTS_SUMMARY_SELECTION_PATCH_MARKER, True)
    results._best_summary_row = _best_summary_row


def _duplicate_metric_columns(metric_columns: Sequence[str]) -> list[str]:
    duplicates: list[str] = []
    seen: set[str] = set()
    for column in metric_columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def _install_results_metric_selection_patch() -> None:
    import neureptrace.results as results

    original_selected_metric_columns = results._selected_metric_columns
    if getattr(original_selected_metric_columns, _RESULTS_METRIC_SELECTION_PATCH_MARKER, False):
        return

    @wraps(original_selected_metric_columns)
    def _selected_metric_columns(
        metric_columns: Sequence[str] | str | None = None,
    ) -> list[str]:
        selected = original_selected_metric_columns(metric_columns)
        duplicates = _duplicate_metric_columns(selected)
        if duplicates:
            raise ValueError(f"metric_columns must not contain duplicates: {duplicates}")
        return selected

    setattr(_selected_metric_columns, _RESULTS_METRIC_SELECTION_PATCH_MARKER, True)
    results._selected_metric_columns = _selected_metric_columns


def install() -> None:
    """Install finite-value and positional-selection report guards."""

    _install_report_patch()
    _install_semantic_stage_patch()
    _install_results_summary_selection_patch()
    _install_results_metric_selection_patch()


__all__ = ["install"]
