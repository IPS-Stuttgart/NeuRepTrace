"""Apply OpenNeuro real-vs-shuffle and diagnostics runtime patches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_TIME_SELECTION_PATCH_MARKER = "_neureptrace_openneuro_real_shuffle_time_selection_patch_installed"
_MANIFEST_BOOL_PATCH_MARKER = "_neureptrace_openneuro_manifest_bool_token_patch_installed"
_PROVENANCE_VALUE_PATCH_MARKER = "_neureptrace_openneuro_provenance_value_patch_installed"
_ALIGNMENT_COMPARE_FIRST_NONEMPTY_PATCH_MARKER = "_neureptrace_openneuro_alignment_compare_first_nonempty_patch_installed"

BOOLEAN_MANIFEST_COMPATIBILITY_COLUMNS = {
    "run_decode",
    "skip_failed_subjects",
    "temporal_smoothing",
    "response_window_ensemble",
    "ensemble_source_baseline_debiasing",
    "label_shuffle_control",
}
_TRUE_TOKENS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "f", "no", "n", "off"}


def _bool_token(value: Any) -> str | None:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric == 0.0:
            return "false"
        if numeric == 1.0:
            return "true"
    text = str(value).strip().lower()
    if text in _TRUE_TOKENS:
        return "true"
    if text in _FALSE_TOKENS:
        return "false"
    return None


def _finite_positions(frame: pd.DataFrame, *columns: str) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values)
    return np.flatnonzero(mask)


def _install_manifest_bool_token_patch() -> None:
    module = importlib.import_module("neureptrace.openneuro_decode_diagnostics")
    original = module._manifest_compatibility_token
    if getattr(original, _MANIFEST_BOOL_PATCH_MARKER, False):
        return

    def _manifest_compatibility_token(column: str, value: Any) -> str:
        if column in BOOLEAN_MANIFEST_COMPATIBILITY_COLUMNS:
            token = _bool_token(value)
            if token is not None:
                return token
        return original(column, value)

    setattr(_manifest_compatibility_token, _MANIFEST_BOOL_PATCH_MARKER, True)
    module._manifest_compatibility_token = _manifest_compatibility_token


def _is_missing_provenance_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _install_provenance_value_patch() -> None:
    module = importlib.import_module("neureptrace.openneuro_decode_diagnostics")
    original = module._provenance_value
    if getattr(original, _PROVENANCE_VALUE_PATCH_MARKER, False):
        return

    def _provenance_value(
        manifest: Mapping[str, Any],
        summary_provenance: Mapping[str, str],
        manifest_key: str,
        summary_key: str | None = None,
    ) -> Any:
        value = manifest.get(manifest_key, "")
        if not _is_missing_provenance_value(value):
            return value
        return summary_provenance.get(summary_key or manifest_key, "")

    setattr(_provenance_value, _PROVENANCE_VALUE_PATCH_MARKER, True)
    module._provenance_value = _provenance_value


def _install_alignment_compare_first_nonempty_patch() -> None:
    module = importlib.import_module("neureptrace.openneuro_alignment_compare")
    original = module._first_nonempty
    if getattr(original, _ALIGNMENT_COMPARE_FIRST_NONEMPTY_PATCH_MARKER, False):
        return

    def _first_nonempty(*values: Any) -> str:
        for value in values:
            if _is_missing_provenance_value(value):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    setattr(_first_nonempty, _ALIGNMENT_COMPARE_FIRST_NONEMPTY_PATCH_MARKER, True)
    module._first_nonempty = _first_nonempty


def _install_time_selection_patch() -> None:
    module = importlib.import_module("neureptrace.openneuro_real_shuffle_report")
    original_nearest_row = module._nearest_row
    if getattr(original_nearest_row, _TIME_SELECTION_PATCH_MARKER, False):
        return

    def _nearest_row(frame: pd.DataFrame, time: float) -> pd.Series:
        if frame.empty or "time" not in frame.columns:
            raise ValueError("Time-course table must contain at least one time row.")
        positions = _finite_positions(frame, "time")
        if positions.size == 0:
            raise ValueError("Time-course table must contain at least one finite time row.")
        times = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
        nearest_position = positions[int(np.argmin(np.abs(times[positions] - float(time))))]
        return frame.iloc[int(nearest_position)]

    def _best_time_row(frame: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.Series:
        if frame.empty or metric not in frame.columns:
            raise ValueError(f"Time-course table must contain '{metric}'.")
        if "time" not in frame.columns:
            raise ValueError("Time-course table must contain 'time'.")
        positions = _finite_positions(frame, "time", metric)
        if positions.size == 0:
            raise ValueError(f"Time-course table must contain at least one finite '{metric}' row with finite time.")
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        best_position = positions[int(np.argmax(values[positions]))]
        return frame.iloc[int(best_position)]

    setattr(_nearest_row, _TIME_SELECTION_PATCH_MARKER, True)
    setattr(_best_time_row, _TIME_SELECTION_PATCH_MARKER, True)
    module._nearest_row = _nearest_row
    module._best_time_row = _best_time_row


def install() -> None:
    """Install OpenNeuro diagnostics compatibility patches."""

    _install_manifest_bool_token_patch()
    _install_provenance_value_patch()
    _install_alignment_compare_first_nonempty_patch()
    _install_time_selection_patch()


__all__ = ["install"]
