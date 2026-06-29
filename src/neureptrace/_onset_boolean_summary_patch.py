"""Robust boolean parsing and missing-group preservation for onset tables.

Onset events and thresholded observations are commonly written to CSV before
being summarized again. Pandas may preserve boolean-looking columns as strings
in user code, and ``Series.astype(bool)`` treats every non-empty string,
including ``"False"``, as true. This patch keeps CSV round-trips from inflating
onset, false-alarm, and threshold-crossing counts.

The onset helpers also group by optional metadata columns such as ``subject``,
``decoder``, and ``emission_mode``. Pandas ``groupby`` drops ``NaN`` keys by
default, which can silently remove whole sequences whose metadata is missing.
This patch maps missing group keys to a private sentinel while the original
helper runs, then restores missing values in the returned table.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np
import pandas as pd

_TRUE_BOOL_TEXT = {"1", "true", "t", "yes", "y", "on"}
_FALSE_BOOL_TEXT = {"0", "false", "f", "no", "n", "off", ""}
_PATCH_MARKER = "_neureptrace_onset_boolean_summary_patch_installed"
_PARSED_ABOVE_THRESHOLD_COLUMN = "_neureptrace_parsed_above_threshold"
_MISSING_GROUP_SENTINEL = object()
_F = TypeVar("_F", bound=Callable[..., Any])


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_bool(value: object, *, name: str) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        numeric = int(value)
        if numeric in {0, 1}:
            return bool(numeric)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(int(numeric))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_BOOL_TEXT:
            return True
        if text in _FALSE_BOOL_TEXT:
            return False
    raise ValueError(f"{name} must contain boolean values, not {value!r}.")


def _bool_series(values: Any, *, name: str) -> pd.Series:
    """Parse a scalar/Series-like boolean column without string truthiness."""

    series = pd.Series(values, copy=False)
    parsed_values = [_parse_bool(value, name=name) for value in series.to_numpy(dtype=object)]
    return pd.Series(parsed_values, index=series.index, dtype=bool)


def _sentinelize_missing_group_values(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Replace missing onset group keys with a private sentinel."""

    missing_columns = [column for column in group_columns if column in frame.columns and bool(frame[column].isna().any())]
    if not missing_columns:
        return frame, []

    patched = frame.copy()
    for column in missing_columns:
        missing = patched[column].isna()
        patched[column] = patched[column].astype(object)
        patched.loc[missing, column] = _MISSING_GROUP_SENTINEL
    return patched, missing_columns


def _restore_missing_group_values(
    frame: pd.DataFrame,
    missing_columns: list[str],
) -> pd.DataFrame:
    """Restore private sentinel group keys to ``np.nan`` in output tables."""

    if not missing_columns:
        return frame

    restored = frame.copy()
    for column in missing_columns:
        if column not in restored.columns:
            continue
        restored[column] = restored[column].map(lambda value: np.nan if value is _MISSING_GROUP_SENTINEL else value)
    return restored


def _wrap_grouped_dataframe_helper(onset_module: Any, function: _F) -> _F:
    """Run an onset helper without letting pandas drop missing group keys."""

    @wraps(function)
    def wrapper(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        group_columns = onset_module._group_columns(frame)
        patched_frame, missing_columns = _sentinelize_missing_group_values(frame, group_columns)
        result = function(patched_frame, *args, **kwargs)
        return _restore_missing_group_values(result, missing_columns)

    return wrapper  # type: ignore[return-value]


def _sequence_crossing_rate(
    onset_module,
    frame: pd.DataFrame,
    sequence_columns: list[str],
    above_threshold: pd.Series,
) -> tuple[int, float]:
    if frame.empty:
        return 0, np.nan
    parsed_frame = frame.copy()
    parsed_frame[_PARSED_ABOVE_THRESHOLD_COLUMN] = above_threshold.to_numpy(dtype=bool)
    onset_module.validate_unique_sequence_times(parsed_frame, sequence_columns)
    sequence_count = 0
    crossing_count = 0
    grouped = parsed_frame.groupby(sequence_columns, sort=True, dropna=False) if sequence_columns else [((), parsed_frame)]
    for _, sequence_frame in grouped:
        sequence_count += 1
        crossing_count += bool(sequence_frame[_PARSED_ABOVE_THRESHOLD_COLUMN].any())
    return crossing_count, crossing_count / sequence_count if sequence_count else np.nan


def _window_threshold_stats(
    onset_module,
    frame: pd.DataFrame,
    window: tuple[float, float],
    sequence_columns: list[str],
) -> dict[str, float | int]:
    window_frame = frame.loc[onset_module._window_mask(frame, window)]
    above_threshold = _bool_series(window_frame["above_threshold"], name="above_threshold")
    sequence_crossing_count, sequence_crossing_rate = _sequence_crossing_rate(
        onset_module,
        window_frame,
        sequence_columns,
        above_threshold,
    )
    stats = {
        "n_observations": len(window_frame),
        "threshold_crossing_count": int(above_threshold.sum()),
        "threshold_crossing_rate": float(above_threshold.mean()) if len(above_threshold) else np.nan,
        "sequence_crossing_count": int(sequence_crossing_count),
        "sequence_crossing_rate": float(sequence_crossing_rate) if np.isfinite(sequence_crossing_rate) else np.nan,
    }
    if "is_correct" in window_frame.columns:
        correct_crossings = _bool_series(
            window_frame.loc[above_threshold, "is_correct"],
            name="is_correct",
        )
        stats["correct_crossing_count"] = int(correct_crossings.sum())
        stats["correct_crossing_rate"] = float(correct_crossings.mean()) if len(correct_crossings) else np.nan
    return stats


def install() -> None:
    """Install robust boolean parsing and missing-group preservation for onset helpers."""

    import neureptrace.onset_detection as onset_detection

    if getattr(onset_detection, _PATCH_MARKER, False):
        return

    original_summarize_onset_events = onset_detection.summarize_onset_events

    @wraps(original_summarize_onset_events)
    def summarize_onset_events(events: pd.DataFrame) -> pd.DataFrame:
        group_columns = onset_detection._group_columns(events)
        rows = []
        grouped = events.groupby(group_columns, sort=True, dropna=False) if group_columns else [((), events)]
        for keys, group_frame in grouped:
            key_values = keys if isinstance(keys, tuple) else (keys,)
            group_values = dict(zip(group_columns, key_values, strict=True))
            detected = _bool_series(group_frame["detected"], name="detected")
            false_alarm = _bool_series(group_frame["detected_before_zero"], name="detected_before_zero")
            correct = _bool_series(group_frame["is_correct_at_detection"], name="is_correct_at_detection")
            post_detected = detected & ~false_alarm
            latencies = pd.to_numeric(group_frame.loc[post_detected, "detection_latency"], errors="coerce").dropna()
            run_durations = pd.to_numeric(
                group_frame.loc[post_detected, "detection_run_duration"],
                errors="coerce",
            ).dropna()
            run_lengths = pd.to_numeric(
                group_frame.loc[post_detected, "detection_run_length"],
                errors="coerce",
            ).dropna()
            rows.append(
                {
                    **group_values,
                    "n_sequences": len(group_frame),
                    "detected_count": int(detected.sum()),
                    "detected_rate": float(detected.mean()) if len(detected) else np.nan,
                    "false_alarm_count": int(false_alarm.sum()),
                    "false_alarm_rate": float(false_alarm.mean()) if len(false_alarm) else np.nan,
                    "post_zero_detected_count": int(post_detected.sum()),
                    "post_zero_detected_rate": float(post_detected.mean()) if len(post_detected) else np.nan,
                    "correct_detection_count": int((detected & correct).sum()),
                    "correct_detection_rate": float((detected & correct).mean()) if len(correct) else np.nan,
                    "post_detection_latency_mean": float(latencies.mean()) if not latencies.empty else np.nan,
                    "post_detection_latency_median": float(latencies.median()) if not latencies.empty else np.nan,
                    "post_detection_run_duration_mean": float(run_durations.mean()) if not run_durations.empty else np.nan,
                    "post_detection_run_duration_median": float(run_durations.median()) if not run_durations.empty else np.nan,
                    "post_detection_run_length_median": float(run_lengths.median()) if not run_lengths.empty else np.nan,
                    "score_threshold": group_frame["score_threshold"].iloc[0] if "score_threshold" in group_frame else np.nan,
                    "threshold_method": group_frame["threshold_method"].iloc[0] if "threshold_method" in group_frame else "",
                    "threshold_quantile": group_frame["threshold_quantile"].iloc[0] if "threshold_quantile" in group_frame else np.nan,
                    "threshold_window_start": group_frame["threshold_window_start"].iloc[0] if "threshold_window_start" in group_frame else np.nan,
                    "threshold_window_stop": group_frame["threshold_window_stop"].iloc[0] if "threshold_window_stop" in group_frame else np.nan,
                    "min_consecutive": group_frame["min_consecutive"].iloc[0] if "min_consecutive" in group_frame else 1,
                    "min_duration": group_frame["min_duration"].iloc[0] if "min_duration" in group_frame else np.nan,
                    "require_stable_prediction": group_frame["require_stable_prediction"].iloc[0] if "require_stable_prediction" in group_frame else False,
                }
            )
        return pd.DataFrame(rows)

    def patched_window_threshold_stats(
        frame: pd.DataFrame,
        window: tuple[float, float],
        sequence_columns: list[str],
    ) -> dict[str, float | int]:
        return _window_threshold_stats(onset_detection, frame, window, sequence_columns)

    onset_detection._window_threshold_stats = patched_window_threshold_stats
    onset_detection.annotate_threshold_crossings = _wrap_grouped_dataframe_helper(
        onset_detection,
        onset_detection.annotate_threshold_crossings,
    )
    onset_detection.detect_onsets = _wrap_grouped_dataframe_helper(
        onset_detection,
        onset_detection.detect_onsets,
    )
    onset_detection.summarize_onset_events = _wrap_grouped_dataframe_helper(
        onset_detection,
        summarize_onset_events,
    )
    onset_detection.summarize_threshold_crossings = _wrap_grouped_dataframe_helper(
        onset_detection,
        onset_detection.summarize_threshold_crossings,
    )
    setattr(onset_detection, _PATCH_MARKER, True)


__all__ = ["install"]
