"""Keep emission comparison outputs schema-stable when no paired modes exist."""

from __future__ import annotations

import importlib
from functools import wraps

import pandas as pd

_PATCH_MARKER = "_neureptrace_emission_compare_empty_pairs_patch_installed"
_COMPARISON_COLUMNS = [
    "decoder",
    "calibrated_observed_gain",
    "uncalibrated_observed_gain",
    "delta_observed_gain",
    "calibrated_control_margin",
    "uncalibrated_control_margin",
    "delta_control_margin",
    "calibrated_effect_minus_baseline_gain",
    "uncalibrated_effect_minus_baseline_gain",
    "delta_effect_minus_baseline_gain",
    "calibrated_shuffled_time_p",
    "uncalibrated_shuffled_time_p",
    "calibrated_shuffled_label_p",
    "uncalibrated_shuffled_label_p",
    "calibrated_best_stay_probability",
    "uncalibrated_best_stay_probability",
    "preferred_emission_mode",
]


def _has_columns(frame: pd.DataFrame) -> bool:
    return all(column in frame.columns for column in _COMPARISON_COLUMNS)


def install() -> None:
    """Patch compare_emission_modes to return an empty schema for unpaired summaries."""

    emission_compare = importlib.import_module("neureptrace.emission_compare")
    original = emission_compare.compare_emission_modes
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def compare_emission_modes(summary: pd.DataFrame) -> pd.DataFrame:
        try:
            comparison = original(summary)
        except KeyError as exc:
            if exc.args != ("delta_control_margin",):
                raise
            validated = emission_compare._validate_temporal_summary(summary)
            for _, decoder_frame in validated.groupby("decoder", sort=True):
                modes = {mode: frame for mode, frame in decoder_frame.groupby("emission_mode", sort=True)}
                if all(mode in modes for mode in emission_compare.PAIRED_EMISSION_MODES):
                    raise
            return pd.DataFrame(columns=_COMPARISON_COLUMNS)
        if comparison.empty and not _has_columns(comparison):
            return pd.DataFrame(columns=_COMPARISON_COLUMNS)
        return comparison

    setattr(compare_emission_modes, _PATCH_MARKER, True)
    emission_compare.compare_emission_modes = compare_emission_modes


__all__ = ["install"]
