"""Pseudo-continuous onset scanning for epoched prediction traces.

This module bridges epoched decoding outputs and the generic NeuRepTrace onset
threshold/detection code.  It treats each validation epoch/trial as an
independent pseudo-continuous sequence whose time axis is expressed relative to
its known event onset.  The detector never receives the event onset as an input;
it only sees sequence/time/score observations and scores detections after the
fact against time zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neureptrace.onset_detection import (
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    annotate_threshold_crossings,
    detect_onsets,
)


@dataclass(frozen=True)
class EpochedOnsetScanResult:
    """Annotated pseudo-continuous observations and per-epoch onset events."""

    observations: pd.DataFrame
    events: pd.DataFrame


def epoched_prediction_traces_to_observations(
    *,
    scores: Sequence[Sequence[float]] | np.ndarray,
    predicted_labels: Sequence[Sequence[Any]] | np.ndarray,
    times: Sequence[float],
    true_labels: Sequence[Any] | None = None,
    sequence_ids: Sequence[Any] | None = None,
    window_starts: Sequence[float] | None = None,
    window_stops: Sequence[float] | None = None,
    window_size: float | None = None,
    metadata: pd.DataFrame | Mapping[str, Any] | None = None,
    score_column: str = "score",
    time_axis: int = 1,
) -> pd.DataFrame:
    """Build canonical observations from epoched prediction traces.

    Parameters
    ----------
    scores, predicted_labels:
        Two-dimensional arrays with one score/prediction per epoch and scan
        time.  By default the shape is ``(n_epochs, n_times)``.  Set
        ``time_axis=0`` for ``(n_times, n_epochs)`` inputs.
    times:
        Scan-window centers in seconds relative to the epoch onset.
    true_labels, sequence_ids:
        Optional per-epoch labels and stable sequence identifiers.  Sequence
        identifiers default to ``range(n_epochs)``.
    window_starts, window_stops, window_size:
        Window bounds used by persistence/duration-aware onset detectors.  Pass
        explicit starts/stops or a single ``window_size`` from which bounds are
        derived around each time point.
    metadata:
        Optional per-epoch metadata.  A DataFrame must contain one row per
        epoch.  A mapping may contain either scalar values, which are repeated,
        or per-epoch sequences.
    score_column:
        Name of the emitted score column.  Use a semantic name such as
        ``"predicted_class_score"`` when the trace is not a calibrated
        probability.
    time_axis:
        Axis in ``scores`` and ``predicted_labels`` corresponding to time.

    Returns
    -------
    pandas.DataFrame
        Observation rows accepted by :func:`run_epoched_onset_scan` and by the
        generic onset-detection functions.
    """

    score_array = _as_2d_array("scores", scores, dtype=float)
    prediction_array = _as_2d_array("predicted_labels", predicted_labels, dtype=object)
    if score_array.shape != prediction_array.shape:
        raise ValueError("scores and predicted_labels must have the same shape.")
    if time_axis not in {0, 1}:
        raise ValueError("time_axis must be 0 or 1.")
    if time_axis == 0:
        score_array = score_array.T
        prediction_array = prediction_array.T

    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    n_sequences, n_times = score_array.shape
    if len(time_values) != n_times:
        raise ValueError("times length must match the time dimension of scores.")

    sequence_values = list(range(n_sequences)) if sequence_ids is None else list(sequence_ids)
    if len(sequence_values) != n_sequences:
        raise ValueError("sequence_ids length must match the number of epochs.")

    true_values = None if true_labels is None else list(true_labels)
    if true_values is not None and len(true_values) != n_sequences:
        raise ValueError("true_labels length must match the number of epochs.")

    starts, stops = _window_bounds(time_values, window_starts, window_stops, window_size)
    metadata_rows = _metadata_rows(metadata, n_sequences)
    rows: list[dict[str, Any]] = []
    for sequence_position, sequence_id in enumerate(sequence_values):
        sequence_metadata = metadata_rows[sequence_position]
        true_label = _scalar(true_values[sequence_position]) if true_values is not None else None
        for time_position, time_value in enumerate(time_values):
            predicted_label = _scalar(prediction_array[sequence_position, time_position])
            row: dict[str, Any] = {
                **sequence_metadata,
                "sequence_id": _scalar(sequence_id),
                "time": float(time_value),
                "window_start": float(starts[time_position]),
                "window_stop": float(stops[time_position]),
                "predicted_label": predicted_label,
                "predicted_class": str(predicted_label),
                score_column: float(score_array[sequence_position, time_position]),
            }
            if true_values is not None:
                row["true_label"] = true_label
                row["true_class"] = str(true_label)
                row["is_correct"] = bool(predicted_label == true_label)
            rows.append(row)
    return pd.DataFrame(rows)


def standardize_epoched_onset_observations(
    observations: pd.DataFrame,
    *,
    sequence_column: str = "sequence_id",
    time_column: str = "time",
    window_start_column: str = "window_start",
    window_stop_column: str = "window_stop",
    true_label_column: str | None = "true_label",
    predicted_label_column: str | None = "predicted_label",
    true_class_column: str | None = "true_class",
    predicted_class_column: str | None = "predicted_class",
    correct_column: str | None = "is_correct",
    score_column: str = "score",
) -> pd.DataFrame:
    """Return observations with the canonical columns expected by onset detection.

    The function intentionally keeps source-specific columns in place.  Aliased
    columns are copied into the generic names used by ``detect_onsets``; this
    lets downstream projects preserve their legacy CSV schemas while sharing the
    same detection path.
    """

    frame = observations.copy()
    _copy_alias(frame, sequence_column, "sequence_id", required=True)
    _copy_alias(frame, time_column, "time", required=True)
    _copy_alias(frame, window_start_column, "window_start", required=False)
    _copy_alias(frame, window_stop_column, "window_stop", required=False)
    if true_label_column is not None:
        _copy_alias(frame, true_label_column, "true_label", required=False)
    if predicted_label_column is not None:
        _copy_alias(frame, predicted_label_column, "predicted_label", required=False)
    if true_class_column is not None:
        _copy_alias(frame, true_class_column, "true_class", required=False)
    if predicted_class_column is not None:
        _copy_alias(frame, predicted_class_column, "predicted_class", required=False)
    if correct_column is not None:
        _copy_alias(frame, correct_column, "is_correct", required=False)

    if score_column not in frame.columns:
        raise ValueError(f"Observation rows must contain score column '{score_column}'.")
    if "predicted_label" not in frame.columns and "predicted_class" not in frame.columns:
        raise ValueError("Observation rows must contain predicted_label or predicted_class.")
    if "predicted_label" in frame.columns and "predicted_class" not in frame.columns:
        frame["predicted_class"] = frame["predicted_label"].astype(str)
    if "predicted_class" in frame.columns and "predicted_label" not in frame.columns:
        frame["predicted_label"] = frame["predicted_class"]
    if "true_label" in frame.columns and "true_class" not in frame.columns:
        frame["true_class"] = frame["true_label"].astype(str)
    if "true_class" in frame.columns and "true_label" not in frame.columns:
        frame["true_label"] = frame["true_class"]
    if {"true_label", "predicted_label"}.issubset(frame.columns) and "is_correct" not in frame.columns:
        frame["is_correct"] = frame["true_label"].astype(object) == frame["predicted_label"].astype(object)
    return frame


def annotate_epoched_onset_scan(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "score",
    threshold_method: str = "point",
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
    **standardize_kwargs: Any,
) -> pd.DataFrame:
    """Annotate epoched scan observations with baseline-derived thresholds."""

    standardized = standardize_epoched_onset_observations(
        observations,
        score_column=score_column,
        **standardize_kwargs,
    )
    return annotate_threshold_crossings(
        standardized,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )


def detect_epoched_onset_events(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "score",
    threshold_method: str = "point",
    detection_start: float | None = None,
    detection_window: tuple[float, float] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
    **standardize_kwargs: Any,
) -> pd.DataFrame:
    """Detect first threshold crossings in epoched pseudo-continuous traces."""

    standardized = standardize_epoched_onset_observations(
        observations,
        score_column=score_column,
        **standardize_kwargs,
    )
    return detect_onsets(
        standardized,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        detection_start=detection_start,
        detection_window=detection_window,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )


def run_epoched_onset_scan(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "score",
    threshold_method: str = "point",
    detection_start: float | None = None,
    detection_window: tuple[float, float] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
    **standardize_kwargs: Any,
) -> EpochedOnsetScanResult:
    """Annotate and score an epoched pseudo-continuous onset scan.

    This is the preferred adapter for dataset-specific projects.  They only need
    to provide sequence/time/score rows; thresholding and first-event extraction
    then share the same path used for continuous probability observations.
    """

    thresholded = annotate_epoched_onset_scan(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
        **standardize_kwargs,
    )
    events = detect_onsets(
        thresholded,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        detection_start=detection_start,
        detection_window=detection_window,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    return EpochedOnsetScanResult(observations=thresholded, events=events)


def _as_2d_array(name: str, values: Sequence[Sequence[Any]] | np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    return array


def _window_bounds(
    times: np.ndarray,
    starts: Sequence[float] | None,
    stops: Sequence[float] | None,
    window_size: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if starts is not None or stops is not None:
        if starts is None or stops is None:
            raise ValueError("window_starts and window_stops must be provided together.")
        start_values = np.asarray(starts, dtype=float)
        stop_values = np.asarray(stops, dtype=float)
    elif window_size is not None:
        if window_size < 0:
            raise ValueError("window_size must be non-negative.")
        start_values = times - window_size / 2.0
        stop_values = times + window_size / 2.0
    else:
        start_values = np.full(len(times), np.nan, dtype=float)
        stop_values = np.full(len(times), np.nan, dtype=float)
    if start_values.shape != times.shape or stop_values.shape != times.shape:
        raise ValueError("window bounds must match the length of times.")
    return start_values, stop_values


def _metadata_rows(metadata: pd.DataFrame | Mapping[str, Any] | None, n_sequences: int) -> list[dict[str, Any]]:
    if metadata is None:
        return [{} for _ in range(n_sequences)]
    if isinstance(metadata, pd.DataFrame):
        if len(metadata) != n_sequences:
            raise ValueError("metadata DataFrame must contain one row per epoch.")
        return metadata.reset_index(drop=True).to_dict(orient="records")

    rows = [dict() for _ in range(n_sequences)]
    for column, value in metadata.items():
        if _is_per_sequence_value(value, n_sequences):
            values = list(value)  # type: ignore[arg-type]
            for row, item in zip(rows, values, strict=True):
                row[column] = _scalar(item)
        else:
            scalar = _scalar(value)
            for row in rows:
                row[column] = scalar
    return rows


def _is_per_sequence_value(value: Any, n_sequences: int) -> bool:
    if isinstance(value, str):
        return False
    try:
        return len(value) == n_sequences  # type: ignore[arg-type]
    except TypeError:
        return False


def _copy_alias(frame: pd.DataFrame, source: str, target: str, *, required: bool) -> None:
    if source in frame.columns:
        if source != target:
            frame[target] = frame[source]
        return
    if target in frame.columns:
        return
    if required:
        raise ValueError(f"Observation rows must contain '{source}' or '{target}'.")


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "EpochedOnsetScanResult",
    "annotate_epoched_onset_scan",
    "detect_epoched_onset_events",
    "epoched_prediction_traces_to_observations",
    "run_epoched_onset_scan",
    "standardize_epoched_onset_observations",
]
