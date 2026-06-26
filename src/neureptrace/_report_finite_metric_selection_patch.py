"""Runtime patch for finite metric selection in time-decoding reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    return values.where(finite)


def _window_mean(frame: pd.DataFrame, column: str, start: float, stop: float) -> float:
    window = frame[(frame["time"] >= start) & (frame["time"] <= stop)]
    if window.empty:
        raise ValueError(f"No time points found in window [{start}, {stop}].")

    values = _finite_numeric_values(window, column).dropna()
    if values.empty:
        raise ValueError(f"No finite values found in column '{column}' for window [{start}, {stop}].")
    return float(values.mean())


def _best_metric_row(frame: pd.DataFrame, selection_metric: str, column: str) -> pd.Series:
    import neureptrace.report as report

    selection_metric = report._validate_selection_metric(selection_metric)
    if column not in frame.columns:
        raise ValueError(f"Frame is missing selection metric column '{column}'.")

    values = _finite_numeric_values(frame, column)
    if values.notna().sum() == 0:
        raise ValueError(f"Selection metric column '{column}' contains no finite values.")

    index = values.idxmax() if report.METRIC_HIGHER_IS_BETTER[selection_metric] else values.idxmin()
    return frame.loc[index]


def install() -> None:
    """Install finite-value guards for report metric selection and window means."""
    import neureptrace.report as report

    if getattr(report._best_metric_row, "_report_finite_metric_selection_patched", False):
        return

    _best_metric_row._report_finite_metric_selection_patched = True  # type: ignore[attr-defined]
    _window_mean._report_finite_metric_selection_patched = True  # type: ignore[attr-defined]
    report._best_metric_row = _best_metric_row
    report._window_mean = _window_mean


__all__ = ["install"]
