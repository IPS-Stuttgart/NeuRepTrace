"""Ignore non-finite LOSO diagnostic best-time candidates."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_loso_diagnostics_finite_time_selection_patch_installed"


def _finite_positions(frame: pd.DataFrame, *columns: str) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values)
    return np.flatnonzero(mask)


def install() -> None:
    """Make LOSO best-time selection ignore rows with non-finite time/metrics."""

    module = importlib.import_module("neureptrace.loso_observation_diagnostics")
    original_best_time = module._best_time
    if getattr(original_best_time, _PATCH_MARKER, False):
        return

    def _best_time(summary: pd.DataFrame, metric: str) -> float:
        if metric not in module.SELECTION_METRICS:
            raise ValueError(f"Unknown selection metric '{metric}'. Available metrics: {', '.join(module.SELECTION_METRICS)}.")
        if summary.empty:
            raise ValueError("Cannot select a best time from an empty summary.")
        if "time" not in summary.columns:
            raise ValueError("Time-course summary must contain 'time'.")
        if metric not in summary.columns:
            raise ValueError(f"Time-course summary must contain '{metric}'.")

        positions = _finite_positions(summary, "time", metric)
        if positions.size == 0:
            raise ValueError(f"Time-course summary must contain at least one finite '{metric}' row with finite time.")

        values = pd.to_numeric(summary[metric], errors="coerce").to_numpy(dtype=float)
        if metric in module.MINIMIZE_METRICS:
            selected_position = positions[int(np.argmin(values[positions]))]
        else:
            selected_position = positions[int(np.argmax(values[positions]))]
        return float(summary.iloc[int(selected_position)]["time"])

    setattr(_best_time, _PATCH_MARKER, True)
    module._best_time = _best_time


__all__ = ["install"]
