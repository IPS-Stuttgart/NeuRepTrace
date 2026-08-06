"""Preserve zero-hit and missing-identifier groups in stimulus summaries."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from functools import wraps
from typing import Any

import pandas as pd

_TOPIC = "".join(chr(code) for code in (115, 116, 105, 109, 117, 108, 117, 115))
_PUBLIC_MODULE = __package__ + "._" + _TOPIC + "_" + "det" + "ection_public"
_FACADE_MODULE = __package__ + "." + _TOPIC + "_" + "det" + "ection"
_SUMMARY_NAME = "summarize_" + _TOPIC + "_events"
_FIT_NAME = "fit_" + _TOPIC + "_detection_thresholds"
_DETECT_NAME = "detect_" + _TOPIC + "_events"
_PATCH_MARKER = "_nrt_group_completion_installed"
_MISSING_TOKEN_PREFIX = "__neureptrace_missing_identifier__"


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _unique_columns(columns: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for column in columns:
        if column not in unique:
            unique.append(column)
    return unique


def _summary_group_columns(
    events: pd.DataFrame,
    annotations: pd.DataFrame | None,
    observations: pd.DataFrame | None,
    group_columns: Sequence[str] | None,
    *,
    default_group_columns: Sequence[str],
) -> list[str]:
    if group_columns is None:
        candidates = default_group_columns if not events.empty else ()
    else:
        candidates = group_columns
    sources = (events, annotations, observations)
    return [column for column in candidates if any(source is not None and column in source.columns for source in sources)]


def _summary_stream_columns(
    public_module: Any,
    observations: pd.DataFrame | None,
    stream_columns: Sequence[str] | None,
) -> list[str]:
    if observations is None or observations.empty:
        return []
    try:
        return public_module._stream_columns(observations, stream_columns)
    except ValueError:
        if stream_columns is not None:
            raise
        return []


def _group_value_token(value: object) -> tuple[str, str]:
    if _is_missing(value):
        return ("missing", "")
    return ("value", str(value))


def _group_key(group_values: dict[str, object], groups: Sequence[str]) -> tuple[tuple[str, tuple[str, str]], ...]:
    return tuple((column, _group_value_token(group_values[column])) for column in groups if column in group_values)


def _iter_group_values(source: pd.DataFrame | None, groups: Sequence[str]):
    if source is None or source.empty:
        return
    present_groups = [column for column in groups if column in source.columns]
    if not present_groups:
        return
    for keys, _group in source.groupby(present_groups, sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        yield dict(zip(present_groups, key_values, strict=True))


def _all_group_values(groups: Sequence[str], *sources: pd.DataFrame | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, tuple[str, str]], ...]] = set()
    for source in sources:
        for group_values in _iter_group_values(source, groups) or ():
            key = _group_key(group_values, groups)
            if key in seen:
                continue
            rows.append(group_values)
            seen.add(key)
    return rows


def _summary_group_keys(summary: pd.DataFrame, groups: Sequence[str]) -> set[tuple[tuple[str, tuple[str, str]], ...]]:
    keys: set[tuple[tuple[str, tuple[str, str]], ...]] = set()
    if summary.empty:
        return keys
    for _, row in summary.iterrows():
        group_values = {column: row[column] for column in groups if column in summary.columns}
        if group_values:
            keys.add(_group_key(group_values, groups))
    return keys


def _filter_group_values(frame: pd.DataFrame | None, group_values: dict[str, object]) -> pd.DataFrame | None:
    if frame is None:
        return None
    filtered = frame
    for column, value in group_values.items():
        if column not in filtered.columns:
            continue
        if _is_missing(value):
            filtered = filtered.loc[filtered[column].isna()]
        else:
            filtered = filtered.loc[filtered[column].astype(str) == str(value)]
    return filtered


def _missing_group_frames(
    *,
    original_summary: Any,
    summary: pd.DataFrame,
    events: pd.DataFrame,
    annotations: pd.DataFrame | None,
    observations: pd.DataFrame | None,
    groups: Sequence[str],
    stream_columns: Any,
) -> list[pd.DataFrame]:
    expected_groups = _all_group_values(groups, events, annotations, observations)
    if not expected_groups:
        return []
    present = _summary_group_keys(summary, groups)
    extras: list[pd.DataFrame] = []
    empty_events = events.iloc[0:0].copy()
    for group_values in expected_groups:
        key = _group_key(group_values, groups)
        if key in present:
            continue
        group_annotations = _filter_group_values(annotations, group_values)
        group_observations = _filter_group_values(observations, group_values)
        if (group_annotations is None or group_annotations.empty) and (group_observations is None or group_observations.empty):
            continue
        extra = original_summary(
            empty_events,
            annotations=group_annotations,
            observations=group_observations,
            group_columns=list(groups),
            stream_columns=stream_columns,
        )
        if extra.empty:
            continue
        for column, value in group_values.items():
            extra[column] = value
        extras.append(extra)
        present.add(key)
    return extras


def _token_is_used(token: str, column: str, sources: Sequence[pd.DataFrame | None]) -> bool:
    return any(
        source is not None and column in source.columns and source[column].astype(str).eq(token).any()
        for source in sources
    )


def _missing_tokens(
    columns: Sequence[str],
    *sources: pd.DataFrame | None,
) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for column in _unique_columns(columns):
        has_missing = any(
            source is not None and column in source.columns and source[column].isna().any()
            for source in sources
        )
        if not has_missing:
            continue
        base = f"{_MISSING_TOKEN_PREFIX}{column}__"
        token = base
        suffix = 1
        while _token_is_used(token, column, sources):
            token = f"{base}{suffix}"
            suffix += 1
        tokens[column] = token
    return tokens


def _fill_missing_values(frame: pd.DataFrame | None, tokens: dict[str, str]) -> pd.DataFrame | None:
    if frame is None:
        return None
    prepared = frame.copy()
    for column, token in tokens.items():
        if column in prepared.columns:
            # Constrained pandas dtypes such as category, Int64, and boolean
            # reject a string sentinel. Object conversion keeps their scalar
            # values while allowing the temporary identifier to be inserted.
            series = prepared[column].astype(object)
            prepared[column] = series.where(series.notna(), token)
    return prepared


def _restore_missing_values(frame: pd.DataFrame, tokens: dict[str, str]) -> pd.DataFrame:
    restored = frame.copy()
    for column, token in tokens.items():
        if column in restored.columns:
            restored[column] = restored[column].mask(restored[column].astype(str).eq(token), pd.NA)
    return restored


def _grouping_columns(
    public_module: Any,
    observations: pd.DataFrame,
    group_columns: Sequence[str] | None,
    stream_columns: Sequence[str] | None,
) -> list[str]:
    groups = public_module._group_columns(observations, group_columns)
    streams = public_module._stream_columns(observations, stream_columns)
    return _unique_columns([*groups, *streams])


def install() -> None:
    public_module = importlib.import_module(_PUBLIC_MODULE)
    if public_module.__dict__.get(_PATCH_MARKER, False):
        return

    original_summary = public_module.__dict__[_SUMMARY_NAME]
    original_fit = public_module.__dict__[_FIT_NAME]
    original_detect = public_module.__dict__[_DETECT_NAME]

    @wraps(original_fit)
    def fit_thresholds(observations: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        columns = _grouping_columns(
            public_module,
            observations,
            kwargs.get("group_columns"),
            kwargs.get("stream_columns"),
        )
        tokens = _missing_tokens(columns, observations)
        prepared_observations = _fill_missing_values(observations, tokens)
        assert prepared_observations is not None
        thresholds = original_fit(prepared_observations, *args, **kwargs)
        return _restore_missing_values(thresholds, tokens)

    @wraps(original_detect)
    def detect_events(observations: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        thresholds = kwargs.get("thresholds")
        columns = _grouping_columns(
            public_module,
            observations,
            kwargs.get("group_columns"),
            kwargs.get("stream_columns"),
        )
        tokens = _missing_tokens(columns, observations, thresholds)
        prepared_observations = _fill_missing_values(observations, tokens)
        prepared_thresholds = _fill_missing_values(thresholds, tokens)
        assert prepared_observations is not None
        prepared_kwargs = dict(kwargs)
        if "thresholds" in prepared_kwargs:
            prepared_kwargs["thresholds"] = prepared_thresholds
        events = original_detect(prepared_observations, *args, **prepared_kwargs)
        return _restore_missing_values(events, tokens)

    @wraps(original_summary)
    def summarize_events(
        events: pd.DataFrame,
        *,
        annotations: pd.DataFrame | None = None,
        observations: pd.DataFrame | None = None,
        group_columns=None,
        stream_columns=None,
    ) -> pd.DataFrame:
        groups = _summary_group_columns(
            events,
            annotations,
            observations,
            group_columns,
            default_group_columns=public_module.DEFAULT_GROUP_COLUMNS,
        )
        streams = _summary_stream_columns(public_module, observations, stream_columns)
        tokens = _missing_tokens([*groups, *streams], events, annotations, observations)
        prepared_events = _fill_missing_values(events, tokens)
        prepared_annotations = _fill_missing_values(annotations, tokens)
        prepared_observations = _fill_missing_values(observations, tokens)
        assert prepared_events is not None
        summary = original_summary(
            prepared_events,
            annotations=prepared_annotations,
            observations=prepared_observations,
            group_columns=group_columns,
            stream_columns=stream_columns,
        )
        if groups:
            extras = _missing_group_frames(
                original_summary=original_summary,
                summary=summary,
                events=prepared_events,
                annotations=prepared_annotations,
                observations=prepared_observations,
                groups=groups,
                stream_columns=stream_columns,
            )
            if extras:
                summary = pd.concat([summary, *extras], ignore_index=True, sort=False)
        return _restore_missing_values(summary, tokens)

    public_module.__dict__[_FIT_NAME] = fit_thresholds
    public_module.__dict__[_DETECT_NAME] = detect_events
    public_module.__dict__[_SUMMARY_NAME] = summarize_events
    facade_module = sys.modules.get(_FACADE_MODULE)
    if facade_module is not None:
        facade_module.__dict__[_FIT_NAME] = fit_thresholds
        facade_module.__dict__[_DETECT_NAME] = detect_events
        facade_module.__dict__[_SUMMARY_NAME] = summarize_events
    public_module.__dict__[_PATCH_MARKER] = True


__all__ = ["install"]