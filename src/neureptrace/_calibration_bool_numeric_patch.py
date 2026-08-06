"""Reject invalid calibration numeric fields before report generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as _calibration

_ORIGINAL_VALIDATE_CALIBRATION_SUMMARY = _calibration._validate_calibration_summary
_ORIGINAL_VALIDATE_RELIABILITY_BINS = _calibration._validate_reliability_bins
_ORIGINAL_VALIDATE_TIME_WINDOW = _calibration._validate_time_window
_ORIGINAL_SUMMARIZE_CALIBRATION_METRICS = _calibration.summarize_calibration_metrics
_INSTALLED = False


def _boolean_rows(values: pd.Series) -> list[int]:
    mask = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    return mask[mask].index.tolist()[:5]


def _reject_boolean_numeric_values(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        bad_rows = _boolean_rows(frame[column])
        if bad_rows:
            raise ValueError(
                f"{source} contains boolean values in numeric column '{column}' at row(s) {bad_rows}."
            )


def _is_complex_scalar(value: object) -> bool:
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 0:
        try:
            scalar = value.item()
        except ValueError:
            return False
        return isinstance(scalar, (complex, np.complexfloating))
    return False


def _patched_validate_time_window(window: Sequence[object], *, name: str) -> tuple[float, float]:
    if not isinstance(window, (str, bytes)):
        try:
            values = tuple(window)
        except TypeError:
            values = ()
        for value in values:
            if _is_complex_scalar(value):
                raise ValueError(f"{name} endpoints must be finite real numeric values, not complex values.")
    return _ORIGINAL_VALIDATE_TIME_WINDOW(window, name=name)


def _patched_validate_calibration_summary(summary: pd.DataFrame) -> pd.DataFrame:
    _reject_boolean_numeric_values(
        summary,
        _calibration.SUMMARY_NUMERIC_COLUMNS,
        source="Summary",
    )
    validated = _ORIGINAL_VALIDATE_CALIBRATION_SUMMARY(summary)
    negative_brier = validated["brier_mean"] < 0.0
    if negative_brier.any():
        bad_rows = negative_brier[negative_brier].index.tolist()[:5]
        raise ValueError(f"Summary contains negative brier_mean at row(s) {bad_rows}.")
    return validated


def _patched_validate_reliability_bins(frame: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    _reject_boolean_numeric_values(
        frame,
        _calibration.RELIABILITY_BIN_NUMERIC_COLUMNS,
        source=str(csv_path),
    )
    _reject_boolean_numeric_values(
        frame,
        (_calibration.RELIABILITY_BIN_WEIGHT_COLUMN,),
        source=str(csv_path),
    )
    return _ORIGINAL_VALIDATE_RELIABILITY_BINS(frame, csv_path)


@wraps(_ORIGINAL_SUMMARIZE_CALIBRATION_METRICS)
def _patched_summarize_calibration_metrics(
    summary: pd.DataFrame,
    *,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> pd.DataFrame:
    """Make best-ECE row selection independent of caller-provided index labels."""

    if not summary.index.is_unique:
        summary = summary.reset_index(drop=True)
    return _ORIGINAL_SUMMARIZE_CALIBRATION_METRICS(
        summary,
        baseline_window=baseline_window,
        effect_window=effect_window,
    )


def install() -> None:
    """Install calibration numeric validation guardrails."""

    global _INSTALLED
    if _INSTALLED:
        return
    _calibration._validate_time_window = _patched_validate_time_window
    _calibration._validate_calibration_summary = _patched_validate_calibration_summary
    _calibration._validate_reliability_bins = _patched_validate_reliability_bins
    _calibration.summarize_calibration_metrics = _patched_summarize_calibration_metrics
    _INSTALLED = True
