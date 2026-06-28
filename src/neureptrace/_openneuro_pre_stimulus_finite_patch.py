"""Ignore non-finite OpenNeuro pre-stimulus summary metric rows."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from . import _openneuro_empty_runs_patch

_PATCH_MARKER = "_neureptrace_openneuro_pre_stimulus_finite_patch_installed"


def _finite_numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _metric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return _finite_numeric(frame[column])


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def install() -> None:
    """Make the OpenNeuro pre-stimulus sanity summary finite-aware."""

    _openneuro_empty_runs_patch.install()

    module = importlib.import_module("neureptrace.openneuro_real_shuffle_report")
    original = module._pre_stimulus_summary
    if getattr(original, _PATCH_MARKER, False):
        return

    def _pre_stimulus_summary(frame: pd.DataFrame) -> dict[str, float | int]:
        if "time" not in frame.columns:
            raise ValueError("Time-course table must contain 'time'.")

        times = _finite_numeric(frame["time"])
        pre_positions = np.flatnonzero(np.isfinite(times) & (times < 0.0))
        if pre_positions.size == 0:
            return {
                "n_pre_stimulus_times": 0,
                "pre_stimulus_balanced_accuracy_mean": float("nan"),
                "pre_stimulus_balanced_accuracy_max": float("nan"),
                "pre_stimulus_balanced_accuracy_max_time": float("nan"),
                "pre_stimulus_top2_accuracy_mean": float("nan"),
                "pre_stimulus_top2_accuracy_max": float("nan"),
            }

        pre = frame.iloc[pre_positions]
        pre_times = times[pre_positions]
        balanced = _metric_values(pre, "balanced_accuracy")
        top2 = _metric_values(pre, "top2_accuracy")
        finite_balanced_positions = np.flatnonzero(np.isfinite(balanced))
        if finite_balanced_positions.size:
            best_local_position = finite_balanced_positions[int(np.argmax(balanced[finite_balanced_positions]))]
            balanced_max_time = float(pre_times[best_local_position])
        else:
            balanced_max_time = float("nan")

        return {
            "n_pre_stimulus_times": int(pre_positions.size),
            "pre_stimulus_balanced_accuracy_mean": _finite_mean(balanced),
            "pre_stimulus_balanced_accuracy_max": _finite_max(balanced),
            "pre_stimulus_balanced_accuracy_max_time": balanced_max_time,
            "pre_stimulus_top2_accuracy_mean": _finite_mean(top2),
            "pre_stimulus_top2_accuracy_max": _finite_max(top2),
        }

    setattr(_pre_stimulus_summary, _PATCH_MARKER, True)
    module._pre_stimulus_summary = _pre_stimulus_summary


__all__ = ["install"]
