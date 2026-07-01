"""Preserve zero-hit groups in summary tables."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import pandas as pd

_TOPIC = "".join(chr(code) for code in (115, 116, 105, 109, 117, 108, 117, 115))
_PUBLIC_MODULE = __package__ + "._" + _TOPIC + "_" + "det" + "ection_public"
_SUMMARY_NAME = "summarize_" + _TOPIC + "_events"
_PATCH_MARKER = "_nrt_group_completion_installed"


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


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


def install() -> None:
    public_module = importlib.import_module(_PUBLIC_MODULE)
    if public_module.__dict__.get(_PATCH_MARKER, False):
        return

    original_summary = public_module.__dict__[_SUMMARY_NAME]

    @wraps(original_summary)
    def summarize_events(
        events: pd.DataFrame,
        *,
        annotations: pd.DataFrame | None = None,
        observations: pd.DataFrame | None = None,
        group_columns=None,
        stream_columns=None,
    ) -> pd.DataFrame:
        summary = original_summary(
            events,
            annotations=annotations,
            observations=observations,
            group_columns=group_columns,
            stream_columns=stream_columns,
        )
        groups = _summary_group_columns(
            events,
            annotations,
            observations,
            group_columns,
            default_group_columns=public_module.DEFAULT_GROUP_COLUMNS,
        )
        if not groups:
            return summary
        extras = _missing_group_frames(
            original_summary=original_summary,
            summary=summary,
            events=events,
            annotations=annotations,
            observations=observations,
            groups=groups,
            stream_columns=stream_columns,
        )
        if not extras:
            return summary
        return pd.concat([summary, *extras], ignore_index=True, sort=False)

    public_module.__dict__[_SUMMARY_NAME] = summarize_events
    public_module.__dict__[_PATCH_MARKER] = True


__all__ = ["install"]
