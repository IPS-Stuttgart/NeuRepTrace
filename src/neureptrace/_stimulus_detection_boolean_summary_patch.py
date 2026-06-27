"""Robust boolean parsing for stimulus-event summary tables.

Stimulus event outputs are often persisted to CSV before downstream summaries
are recomputed.  Pandas or callers may represent boolean columns as strings, and
``Series.astype(bool)`` treats every non-empty string, including ``"False"``, as
true.  This patch prevents CSV round-trips from inflating true-positive and
duplicate-detection counts.
"""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_TRUE_BOOL_TEXT = {"1", "true", "t", "yes", "y", "on"}
_FALSE_BOOL_TEXT = {"0", "false", "f", "no", "n", "off", ""}
_PATCH_MARKER = "_neureptrace_stimulus_boolean_summary_patch_installed"
_BOOLEAN_SUMMARY_COLUMNS = ("is_true_positive", "is_duplicate_detection")


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


def _coerce_boolean_summary_columns(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    parsed = events.copy()
    for column in _BOOLEAN_SUMMARY_COLUMNS:
        if column in parsed.columns:
            parsed[column] = _bool_series(parsed[column], name=column)
    return parsed


def install() -> None:
    """Install robust boolean parsing for stimulus summary helpers."""

    import neureptrace._stimulus_detection_public as stimulus_public

    if getattr(stimulus_public, _PATCH_MARKER, False):
        return

    original_matched_events = stimulus_public._matched_events
    original_summarize_stimulus_events = stimulus_public.summarize_stimulus_events

    def _matched_events(group_frame: pd.DataFrame) -> pd.DataFrame:
        if "is_true_positive" in group_frame.columns:
            parsed_true_positive = _bool_series(group_frame["is_true_positive"], name="is_true_positive")
            return group_frame.loc[parsed_true_positive]
        return original_matched_events(group_frame)

    @wraps(original_summarize_stimulus_events)
    def summarize_stimulus_events(
        events: pd.DataFrame,
        *,
        annotations: pd.DataFrame | None = None,
        observations: pd.DataFrame | None = None,
        group_columns=None,
        stream_columns=None,
    ) -> pd.DataFrame:
        return original_summarize_stimulus_events(
            _coerce_boolean_summary_columns(events),
            annotations=annotations,
            observations=observations,
            group_columns=group_columns,
            stream_columns=stream_columns,
        )

    stimulus_public._matched_events = _matched_events
    stimulus_public.summarize_stimulus_events = summarize_stimulus_events
    setattr(stimulus_public, _PATCH_MARKER, True)

    public_module = sys.modules.get("neureptrace.stimulus_detection")
    if public_module is not None:
        public_module.summarize_stimulus_events = summarize_stimulus_events
        if hasattr(public_module, "_matched_events"):
            public_module._matched_events = _matched_events


__all__ = ["install"]