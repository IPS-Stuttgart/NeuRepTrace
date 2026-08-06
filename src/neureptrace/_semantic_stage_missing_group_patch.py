"""Preserve semantic-stage groups with missing optional identifiers."""

from __future__ import annotations

import importlib
from functools import wraps

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_semantic_stage_missing_group_patch_installed"
_GROUP_COLUMNS = ("decoder", "emission_mode")


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _placeholders(*frames: pd.DataFrame) -> dict[str, str]:
    occupied = {
        str(value)
        for frame in frames
        for value in frame.to_numpy(dtype=object).ravel()
        if not _is_missing_scalar(value)
    }
    result: dict[str, str] = {}
    for column in _GROUP_COLUMNS:
        if not any(column in frame.columns and bool(frame[column].isna().any()) for frame in frames):
            continue
        candidate = f"__neureptrace_missing_{column}__"
        suffix = 1
        while candidate in occupied:
            candidate = f"__neureptrace_missing_{column}_{suffix}__"
            suffix += 1
        result[column] = candidate
        occupied.add(candidate)
    return result


def _protect(frame: pd.DataFrame, placeholders: dict[str, str]) -> pd.DataFrame:
    if not placeholders:
        return frame
    protected = frame.copy()
    for column, placeholder in placeholders.items():
        if column not in protected.columns:
            continue
        protected[column] = protected[column].astype(object)
        protected.loc[protected[column].isna(), column] = placeholder
    return protected


def _restore(frame: pd.DataFrame, placeholders: dict[str, str]) -> pd.DataFrame:
    if not placeholders or frame.empty:
        return frame
    restored = frame.copy()
    for column, placeholder in placeholders.items():
        if column not in restored.columns:
            continue
        mask = restored[column].map(lambda value: isinstance(value, str) and value == placeholder)
        restored.loc[mask, column] = pd.NA
    return restored


def install() -> None:
    importlib.import_module(
        "neureptrace._semantic_stage_reader_missing_group_patch"
    ).install()
    semantic_stages = importlib.import_module("neureptrace.semantic_stages")

    original_dominant = semantic_stages.summarize_dominant_timecourse
    if not getattr(original_dominant, _PATCH_MARKER, False):

        @wraps(original_dominant)
        def summarize_dominant_timecourse(state_traces: pd.DataFrame) -> pd.DataFrame:
            placeholders = _placeholders(state_traces)
            summary = original_dominant(_protect(state_traces, placeholders))
            return _restore(summary, placeholders)

        setattr(summarize_dominant_timecourse, _PATCH_MARKER, True)
        semantic_stages.summarize_dominant_timecourse = summarize_dominant_timecourse

    original_category = semantic_stages.summarize_category_timecourse
    if not getattr(original_category, _PATCH_MARKER, False):

        @wraps(original_category)
        def summarize_category_timecourse(state_traces: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
            placeholders = _placeholders(state_traces)
            summary, state_names = original_category(_protect(state_traces, placeholders))
            return _restore(summary, placeholders), state_names

        setattr(summarize_category_timecourse, _PATCH_MARKER, True)
        semantic_stages.summarize_category_timecourse = summarize_category_timecourse

    original_detect = semantic_stages.detect_stable_stages
    if not getattr(original_detect, _PATCH_MARKER, False):

        @wraps(original_detect)
        def detect_stable_stages(
            time_summary: pd.DataFrame,
            *,
            posterior_threshold: float = 0.6,
            match_threshold: float = 0.6,
            min_duration: float = 0.04,
        ) -> pd.DataFrame:
            placeholders = _placeholders(time_summary)
            stages = original_detect(
                _protect(time_summary, placeholders),
                posterior_threshold=posterior_threshold,
                match_threshold=match_threshold,
                min_duration=min_duration,
            )
            return _restore(stages, placeholders)

        setattr(detect_stable_stages, _PATCH_MARKER, True)
        semantic_stages.detect_stable_stages = detect_stable_stages

    original_report = semantic_stages.build_stage_report
    if not getattr(original_report, _PATCH_MARKER, False):

        @wraps(original_report)
        def build_stage_report(
            time_summary: pd.DataFrame,
            stages: pd.DataFrame,
            *,
            posterior_threshold: float,
            match_threshold: float,
            min_duration: float,
        ) -> str:
            placeholders = _placeholders(time_summary, stages)
            report = original_report(
                _protect(time_summary, placeholders),
                _protect(stages, placeholders),
                posterior_threshold=posterior_threshold,
                match_threshold=match_threshold,
                min_duration=min_duration,
            )
            for placeholder in placeholders.values():
                report = report.replace(placeholder, "")
            return report

        setattr(build_stage_report, _PATCH_MARKER, True)
        semantic_stages.build_stage_report = build_stage_report


__all__ = ["install"]
