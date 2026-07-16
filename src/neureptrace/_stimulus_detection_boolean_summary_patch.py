"""Robust stimulus-event summary parsing and online label inference.

Stimulus event outputs are often persisted to CSV before downstream summaries
are recomputed. Pandas or callers may represent boolean columns as strings, and
``Series.astype(bool)`` treats every non-empty string, including ``"False"``, as
true. This patch prevents CSV round-trips from inflating true-positive and
duplicate-detection counts.

The online detector also needs to infer predictions from ``prob_class_*``
columns when explicit prediction fields are absent. Public class labels can be
signed or non-contiguous, so probability positions must not be used as labels.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_TRUE_BOOL_TEXT = {"1", "true", "t", "yes", "y", "on"}
_FALSE_BOOL_TEXT = {"0", "false", "f", "no", "n", "off", ""}
_PATCH_MARKER = "_neureptrace_stimulus_boolean_summary_patch_installed"
_STREAMING_LABEL_PATCH_MARKER = "_neureptrace_streaming_probability_label_patch_installed"
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


def _integer_probability_suffix(suffix: str) -> int | None:
    text = str(suffix)
    digits = text[1:] if text[:1] in {"+", "-"} else text
    if not digits or not digits.isdigit():
        return None
    return int(text)


def _probability_sort_key(column: str) -> tuple[int, int, str]:
    suffix = str(column).removeprefix("prob_class_")
    label = _integer_probability_suffix(suffix)
    if label is None:
        return 1, 0, suffix
    return 0, label, suffix


def _probability_label_values(columns: Sequence[str]) -> tuple[int, ...]:
    labels = tuple(
        _integer_probability_suffix(str(column).removeprefix("prob_class_"))
        for column in columns
    )
    if any(label is None for label in labels):
        return tuple(range(len(columns)))
    return tuple(int(label) for label in labels if label is not None)


def _duplicate_labels(labels: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    duplicates: list[int] = []
    for label in labels:
        if label in seen and label not in duplicates:
            duplicates.append(label)
        seen.add(label)
    return duplicates


def _streaming_probability_columns(observation: Mapping[str, object]) -> list[str]:
    columns = sorted(
        (str(column) for column in observation if str(column).startswith("prob_class_")),
        key=_probability_sort_key,
    )
    if not columns:
        raise ValueError("Observation rows must contain probability columns named 'prob_class_*'.")
    labels = _probability_label_values(columns)
    if len(labels) == len(columns):
        parsed = tuple(
            _integer_probability_suffix(column.removeprefix("prob_class_"))
            for column in columns
        )
        if all(label is not None for label in parsed):
            duplicates = _duplicate_labels(labels)
            if duplicates:
                raise ValueError(
                    "prob_class_* columns must map to unique class labels; "
                    f"duplicate label(s): {duplicates}."
                )
    return columns


def _has_nonmissing_value(observation: Mapping[str, object], column: str) -> bool:
    if column not in observation:
        return False
    try:
        return not bool(pd.isna(observation[column]))
    except (TypeError, ValueError):
        return True


def _install_streaming_probability_label_patch() -> None:
    """Preserve public labels when the online detector infers predictions."""

    import neureptrace.streaming_stimulus_detection as streaming

    if getattr(streaming, _STREAMING_LABEL_PATCH_MARKER, False):
        return

    original_score_observation = streaming._score_observation

    @wraps(original_score_observation)
    def _score_observation(observation: Mapping[str, object], threshold_row: pd.Series) -> float:
        if str(threshold_row["score_mode"]) != "predicted_class_confidence":
            return original_score_observation(observation, threshold_row)
        if _has_nonmissing_value(observation, "predicted_label") or _has_nonmissing_value(observation, "predicted_class"):
            return original_score_observation(observation, threshold_row)

        confidence = streaming._validated_confidence(observation)
        columns, probabilities = streaming._validated_probability_observation(observation)
        label_values = _probability_label_values(columns)
        predicted_label = label_values[int(probabilities.argmax())]
        return confidence if str(predicted_label) == str(threshold_row["stimulus_label"]) else 0.0

    streaming._probability_columns_from_observation = _streaming_probability_columns
    streaming._score_observation = _score_observation
    setattr(streaming, _STREAMING_LABEL_PATCH_MARKER, True)


def _install_group_completion_patch() -> None:
    """Keep zero-detection group preservation active for direct public-module users."""

    from neureptrace import _group_completion_patch

    _group_completion_patch.install()


def _sync_public_module(stimulus_public: Any) -> None:
    public_module = sys.modules.get("neureptrace.stimulus_detection")
    if public_module is not None:
        public_module.summarize_stimulus_events = stimulus_public.summarize_stimulus_events
        if hasattr(public_module, "_matched_events"):
            public_module._matched_events = stimulus_public._matched_events


def install() -> None:
    """Install robust stimulus summary and streaming-label handling."""

    import neureptrace._stimulus_detection_public as stimulus_public
    from neureptrace import _matched_filter_group_keys_patch, _matched_filter_template_offsets_patch

    _install_streaming_probability_label_patch()
    _matched_filter_template_offsets_patch.install()
    _matched_filter_group_keys_patch.install()

    if getattr(stimulus_public, _PATCH_MARKER, False):
        _install_group_completion_patch()
        _sync_public_module(stimulus_public)
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

    _install_group_completion_patch()
    _sync_public_module(stimulus_public)


__all__ = ["install"]
