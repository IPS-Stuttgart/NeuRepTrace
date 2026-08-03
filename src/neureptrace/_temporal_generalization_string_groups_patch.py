"""Normalize temporal grouping, boolean metadata, and continuous scan controls."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_temporal_generalization_string_groups_patch_installed"
_CONTINUOUS_SEQUENCE_PATCH_MARKER = "_neureptrace_continuous_stream_sequence_identity_patch_installed"
_CONTINUOUS_SLICE_PATCH_MARKER = "_neureptrace_continuous_slice_selection_patch_installed"
_CONTINUOUS_SCAN_STEP_PATCH_MARKER = "_neureptrace_continuous_scan_step_patch_installed"
_CONTINUOUS_ANNOTATION_PATCH_MARKER = "_neureptrace_continuous_annotation_boundary_patch_installed"
_TRUE_BOOL_TEXT = {"1", "true", "t", "yes", "y", "on"}
_FALSE_BOOL_TEXT = {"0", "false", "f", "no", "n", "off", ""}


def _normalize_group_columns(group_columns: Sequence[str] | str | None) -> list[str]:
    if group_columns is None:
        return []
    if isinstance(group_columns, str):
        return [group_columns]
    return list(dict.fromkeys(group_columns))


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
    series = pd.Series(values, copy=False)
    parsed_values = [_parse_bool(value, name=name) for value in series.to_numpy(dtype=object)]
    return pd.Series(parsed_values, index=series.index, dtype=bool)


def _coerce_is_diagonal(frame: Any) -> Any:
    if not isinstance(frame, pd.DataFrame) or "is_diagonal" not in frame.columns:
        return frame
    coerced = frame.copy()
    coerced["is_diagonal"] = _bool_series(coerced["is_diagonal"], name="is_diagonal")
    return coerced


def _scalar_value(value: object, *, name: str) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar value.")
        return value.item()
    return value


def _positive_slice_count(value: object) -> int:
    value = _scalar_value(value, name="slice_count")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("slice_count must be a positive integer.")
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("slice_count must be a positive integer.")
        parsed = int(numeric)
    else:
        raise ValueError("slice_count must be a positive integer.")
    if parsed <= 0:
        raise ValueError("slice_count must be a positive integer.")
    return parsed


def _positive_slice_duration(value: object) -> float:
    value = _scalar_value(value, name="slice_duration")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("slice_duration must be a positive finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("slice_duration must be a positive finite number.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("slice_duration must be a positive finite number.")
    return parsed


def _positive_scan_step(value: object) -> float:
    value = _scalar_value(value, name="scan_step")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("scan_step must be a positive finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("scan_step must be a positive finite number.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("scan_step must be a positive finite number.")
    return parsed


def _finite_slice_starts(values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("slice_starts must contain at least one finite numeric start.")
    try:
        raw_starts = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("slice_starts must contain at least one finite numeric start.") from exc
    if not raw_starts:
        raise ValueError("slice_starts must contain at least one finite numeric start.")

    starts: list[float] = []
    for value in raw_starts:
        value = _scalar_value(value, name="slice_starts")
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("slice_starts must contain only finite numeric starts.")
        try:
            start = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("slice_starts must contain only finite numeric starts.") from exc
        if not np.isfinite(start):
            raise ValueError("slice_starts must contain only finite numeric starts.")
        starts.append(start)
    return tuple(starts)


def _with_scan_stream_sequence_ids(observations: Any) -> Any:
    """Use one generated sequence identifier per scanned continuous stream."""

    if not isinstance(observations, pd.DataFrame):
        return observations
    if "sequence_id" in observations.columns or "stream_id" not in observations.columns:
        return observations
    standardized = observations.copy()
    standardized["sequence_id"] = standardized["stream_id"]
    return standardized


def _scan_standardizer(original):
    @wraps(original)
    def standardize(observations, *args, **kwargs):
        return original(_with_scan_stream_sequence_ids(observations), *args, **kwargs)

    setattr(standardize, _CONTINUOUS_SEQUENCE_PATCH_MARKER, True)
    return standardize


def _scan_segment_builder(original):
    @wraps(original)
    def build_scan_segments(*args, **kwargs):
        slice_duration = kwargs.get("slice_duration")
        slice_starts = kwargs.get("slice_starts")
        slice_count = kwargs.get("slice_count")

        if slice_duration is not None:
            slice_duration = _positive_slice_duration(slice_duration)
            kwargs["slice_duration"] = slice_duration
        if slice_starts is not None:
            slice_starts = _finite_slice_starts(slice_starts)
            kwargs["slice_starts"] = slice_starts
        if slice_count is not None:
            slice_count = _positive_slice_count(slice_count)
            kwargs["slice_count"] = slice_count

        if slice_starts is not None and slice_count is not None:
            raise ValueError("slice_starts and slice_count are mutually exclusive.")
        if slice_duration is None and (slice_starts is not None or slice_count is not None):
            raise ValueError("slice_duration must be provided when slice_starts or slice_count is set.")
        return original(*args, **kwargs)

    setattr(build_scan_segments, _CONTINUOUS_SLICE_PATCH_MARKER, True)
    return build_scan_segments


def _scan_runner(original):
    @wraps(original)
    def run_continuous_stimulus_scan(*args, **kwargs):
        if "scan_step" in kwargs:
            kwargs["scan_step"] = _positive_scan_step(kwargs["scan_step"])
        return original(*args, **kwargs)

    setattr(run_continuous_stimulus_scan, _CONTINUOUS_SCAN_STEP_PATCH_MARKER, True)
    return run_continuous_stimulus_scan


def _same_boundary(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 8.0 * np.finfo(float).eps * scale


def _half_open_annotation_segments(segments: Sequence[Any]) -> tuple[Any, ...]:
    """Make shared slice boundaries belong to the slice that starts there."""

    segment_tuple = tuple(segments)
    starts = tuple(float(segment.start) for segment in segment_tuple)
    adjusted = []
    for index, segment in enumerate(segment_tuple):
        stop = float(segment.stop)
        shared_stop = any(other_index != index and _same_boundary(stop, start) for other_index, start in enumerate(starts))
        adjusted.append(replace(segment, stop=np.nextafter(stop, -np.inf)) if shared_stop else segment)
    return tuple(adjusted)


def _scan_annotation_builder(original):
    @wraps(original)
    def annotation_table(*args, **kwargs):
        segments = kwargs.get("segments")
        if segments is not None:
            kwargs["segments"] = _half_open_annotation_segments(segments)
        return original(*args, **kwargs)

    setattr(annotation_table, _CONTINUOUS_ANNOTATION_PATCH_MARKER, True)
    return annotation_table


def _install_continuous_scan_patches() -> None:
    """Keep scan windows grouped and validate continuous scan controls."""

    module = importlib.import_module("neureptrace.continuous_stimulus_scan")

    original_standardizer = module._standardize_stream_observations
    if not getattr(original_standardizer, _CONTINUOUS_SEQUENCE_PATCH_MARKER, False):
        module._standardize_stream_observations = _scan_standardizer(original_standardizer)

    original_builder = module.build_scan_segments
    if not getattr(original_builder, _CONTINUOUS_SLICE_PATCH_MARKER, False):
        module.build_scan_segments = _scan_segment_builder(original_builder)

    original_runner = module.run_continuous_stimulus_scan
    if not getattr(original_runner, _CONTINUOUS_SCAN_STEP_PATCH_MARKER, False):
        module.run_continuous_stimulus_scan = _scan_runner(original_runner)

    original_annotation_table = module._annotation_table
    if not getattr(original_annotation_table, _CONTINUOUS_ANNOTATION_PATCH_MARKER, False):
        module._annotation_table = _scan_annotation_builder(original_annotation_table)


def install() -> None:
    """Patch temporal grouping, scan identities, scan controls, and CSV booleans."""

    _install_continuous_scan_patches()

    temporal_generalization = importlib.import_module("neureptrace.decoding.temporal_generalization")
    original_summarize = temporal_generalization.summarize_temporal_generalization_matrix
    if getattr(original_summarize, _PATCH_MARKER, False):
        return

    @wraps(original_summarize)
    def summarize_temporal_generalization_matrix(
        frame: Any,
        *,
        group_columns: Sequence[str] | str | None = (),
        accuracy_column: str = "accuracy",
        chance_column: str | None = "chance_accuracy",
    ):
        return original_summarize(
            _coerce_is_diagonal(frame),
            group_columns=_normalize_group_columns(group_columns),
            accuracy_column=accuracy_column,
            chance_column=chance_column,
        )

    setattr(summarize_temporal_generalization_matrix, _PATCH_MARKER, True)
    temporal_generalization.summarize_temporal_generalization_matrix = summarize_temporal_generalization_matrix


__all__ = ["install"]
