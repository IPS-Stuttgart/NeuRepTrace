"""Validate stimulus annotations and match them with an optimal assignment."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Sequence
from functools import wraps

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_PUBLIC_MODULE = f"{__package__}._stimulus_detection_public"
_MATCH_NAME = "match_stimulus_annotations"
_PATCH_MARKER = "_nrt_duplicate_event_index_matching_installed"


def _normalize_match_tolerance(value: object) -> float:
    message = "match_tolerance must be a non-negative finite number."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(message)
    return tolerance


def _validate_onset_times(frame: pd.DataFrame, *, frame_name: str) -> None:
    """Reject onset coordinates that cannot represent finite real times."""

    if "onset_time" not in frame.columns:
        return
    invalid_indices: list[object] = []
    for index, value in frame["onset_time"].items():
        if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
            invalid_indices.append(index)
            continue
        try:
            onset_time = float(value)
        except (TypeError, ValueError, OverflowError):
            invalid_indices.append(index)
            continue
        if not np.isfinite(onset_time):
            invalid_indices.append(index)
    if invalid_indices:
        raise ValueError(
            f"{frame_name} onset_time must contain only finite real numbers; "
            f"invalid row indices: {invalid_indices}."
        )


def _candidate_annotations(
    event: pd.Series,
    annotations: pd.DataFrame,
    *,
    streams: Sequence[str],
    match_tolerance: float,
    require_class_match: bool,
    annotation_id,
    annotation_match_key,
) -> pd.DataFrame:
    candidates = annotations.copy()
    for column in streams:
        if column in candidates.columns:
            candidates = candidates.loc[candidates[column].astype(str) == str(event[column])]
    if require_class_match:
        if "stimulus_class" in candidates.columns:
            candidates = candidates.loc[candidates["stimulus_class"].astype(str) == str(event["stimulus_class"])]
        elif "stimulus_label" in candidates.columns:
            candidates = candidates.loc[candidates["stimulus_label"].astype(str) == str(event["stimulus_label"])]
    if candidates.empty:
        return candidates

    candidates = candidates.copy()
    candidates["_annotation_id"] = [annotation_id(row, index) for index, row in candidates.iterrows()]
    candidates["_annotation_match_key"] = [annotation_match_key(row, streams) for _, row in candidates.iterrows()]
    candidates["_latency"] = float(event["onset_time"]) - pd.to_numeric(candidates["onset_time"], errors="coerce")
    candidates["_abs_latency"] = candidates["_latency"].abs()
    return candidates.loc[
        np.isfinite(candidates["_abs_latency"]) & (candidates["_abs_latency"] <= match_tolerance)
    ].sort_values("_abs_latency", kind="mergesort")


def _optimal_assignment(
    event_indices: Sequence[Hashable],
    candidates_by_event: dict[Hashable, pd.DataFrame],
) -> dict[Hashable, pd.Series]:
    annotation_keys: list[Hashable] = []
    annotation_key_set: set[Hashable] = set()
    best_rows_by_event: dict[Hashable, dict[Hashable, pd.Series]] = {}
    maximum_latency = 0.0

    for event_index in event_indices:
        best_rows: dict[Hashable, pd.Series] = {}
        for _, candidate in candidates_by_event[event_index].iterrows():
            key = candidate["_annotation_match_key"]
            if key in best_rows:
                continue
            best_rows[key] = candidate
            maximum_latency = max(maximum_latency, float(candidate["_abs_latency"]))
            if key not in annotation_key_set:
                annotation_keys.append(key)
                annotation_key_set.add(key)
        best_rows_by_event[event_index] = best_rows

    if not annotation_keys:
        return {}

    n_events = len(event_indices)
    n_annotations = len(annotation_keys)
    key_columns = {key: column for column, key in enumerate(annotation_keys)}
    latency_scale = max(maximum_latency, 1.0)
    unmatched_cost = float(n_events + 1)
    forbidden_cost = 2.0 * unmatched_cost
    costs = np.full((n_events, n_annotations + n_events), forbidden_cost, dtype=float)
    costs[:, n_annotations:] = unmatched_cost

    for event_row, event_index in enumerate(event_indices):
        for key, candidate in best_rows_by_event[event_index].items():
            costs[event_row, key_columns[key]] = float(candidate["_abs_latency"]) / latency_scale

    row_indices, column_indices = linear_sum_assignment(costs)
    assigned: dict[Hashable, pd.Series] = {}
    for event_row, column in zip(row_indices, column_indices, strict=True):
        if column >= n_annotations or costs[event_row, column] >= unmatched_cost:
            continue
        event_index = event_indices[event_row]
        key = annotation_keys[column]
        assigned[event_index] = best_rows_by_event[event_index][key]
    return assigned


def install() -> None:
    public_module = importlib.import_module(_PUBLIC_MODULE)
    if public_module.__dict__.get(_PATCH_MARKER, False):
        return

    required_helpers = (
        _MATCH_NAME,
        "_add_annotation_candidate_columns",
        "_annotation_id",
        "_annotation_match_key",
        "_annotation_value",
        "_stream_columns",
    )
    if not all(name in public_module.__dict__ for name in required_helpers):
        return

    original_match = public_module.__dict__[_MATCH_NAME]
    add_candidate_columns = public_module.__dict__["_add_annotation_candidate_columns"]
    annotation_id = public_module.__dict__["_annotation_id"]
    annotation_match_key = public_module.__dict__["_annotation_match_key"]
    annotation_value = public_module.__dict__["_annotation_value"]
    stream_columns_resolver = public_module.__dict__["_stream_columns"]

    @wraps(original_match)
    def match_stimulus_annotations(
        events: pd.DataFrame,
        annotations: pd.DataFrame,
        *,
        stream_columns: Sequence[str] | None = None,
        match_tolerance: float = 0.1,
        require_class_match: bool = True,
    ) -> pd.DataFrame:
        tolerance = _normalize_match_tolerance(match_tolerance)
        _validate_onset_times(events, frame_name="events")
        original_index = events.index.copy()
        matched = add_candidate_columns(events.reset_index(drop=True))
        if events.empty:
            matched.index = original_index
            return matched

        _validate_onset_times(annotations, frame_name="annotations")
        if "onset_time" not in annotations.columns:
            raise ValueError("Annotation rows must contain onset_time.")
        streams = stream_columns_resolver(matched, stream_columns)
        event_indices = matched.sort_values("onset_time", kind="mergesort").index.tolist()
        candidates_by_event: dict[Hashable, pd.DataFrame] = {}

        for event_index in event_indices:
            event = matched.loc[event_index]
            candidates = _candidate_annotations(
                event,
                annotations,
                streams=streams,
                match_tolerance=tolerance,
                require_class_match=require_class_match,
                annotation_id=annotation_id,
                annotation_match_key=annotation_match_key,
            )
            candidates_by_event[event_index] = candidates
            if candidates.empty:
                continue
            nearest = candidates.iloc[0]
            matched.at[event_index, "candidate_annotation_id"] = nearest["_annotation_id"]
            matched.at[event_index, "candidate_annotation_onset_time"] = float(nearest["onset_time"])
            matched.at[event_index, "candidate_annotation_class"] = annotation_value(nearest, "stimulus_class", default="")
            matched.at[event_index, "candidate_annotation_label"] = annotation_value(nearest, "stimulus_label", default=np.nan)
            matched.at[event_index, "candidate_latency"] = float(nearest["_latency"])

        assigned = _optimal_assignment(event_indices, candidates_by_event)
        for event_index in event_indices:
            annotation = assigned.get(event_index)
            if annotation is None:
                matched.at[event_index, "is_duplicate_detection"] = not candidates_by_event[event_index].empty
                continue
            matched.at[event_index, "matched_annotation_id"] = annotation["_annotation_id"]
            matched.at[event_index, "matched_annotation_onset_time"] = float(annotation["onset_time"])
            matched.at[event_index, "matched_annotation_class"] = annotation_value(annotation, "stimulus_class", default="")
            matched.at[event_index, "matched_annotation_label"] = annotation_value(annotation, "stimulus_label", default=np.nan)
            matched.at[event_index, "latency"] = float(annotation["_latency"])
            matched.at[event_index, "is_true_positive"] = True

        matched.index = original_index
        return matched

    public_module.__dict__[_MATCH_NAME] = match_stimulus_annotations
    public_module.__dict__[_PATCH_MARKER] = True


__all__ = ["install"]
