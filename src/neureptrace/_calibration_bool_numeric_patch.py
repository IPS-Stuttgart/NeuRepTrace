"""Reject boolean-like calibration numeric fields before pandas coercion."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as _calibration

_ORIGINAL_VALIDATE_CALIBRATION_SUMMARY = _calibration._validate_calibration_summary
_ORIGINAL_VALIDATE_RELIABILITY_BINS = _calibration._validate_reliability_bins
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


def _patched_validate_calibration_summary(summary: pd.DataFrame) -> pd.DataFrame:
    _reject_boolean_numeric_values(
        summary,
        _calibration.SUMMARY_NUMERIC_COLUMNS,
        source="Summary",
    )
    return _ORIGINAL_VALIDATE_CALIBRATION_SUMMARY(summary)


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


def install() -> None:
    """Install calibration numeric validation guardrails."""

    global _INSTALLED
    if _INSTALLED:
        return
    _calibration._validate_calibration_summary = _patched_validate_calibration_summary
    _calibration._validate_reliability_bins = _patched_validate_reliability_bins
    _INSTALLED = True
