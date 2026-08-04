"""Validate reaction-time association configuration."""

from __future__ import annotations

import importlib
from functools import wraps
from numbers import Integral

import numpy as np

_PATCH_MARKER = "_neureptrace_reaction_time_analysis_config_patch_installed"


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def install() -> None:
    """Reject invalid association thresholds before numerical analysis."""

    module = importlib.import_module("neureptrace.behavior.reaction_time")
    original = module.analyze_metric_reaction_times
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def analyze_metric_reaction_times(
        rows,
        metrics,
        *,
        reaction_time_column="reaction_time",
        participant_column="participant",
        min_trials=3,
        include_pooled_within_participant=True,
    ):
        return original(
            rows,
            metrics,
            reaction_time_column=reaction_time_column,
            participant_column=participant_column,
            min_trials=_positive_integer(min_trials, name="min_trials"),
            include_pooled_within_participant=include_pooled_within_participant,
        )

    setattr(analyze_metric_reaction_times, _PATCH_MARKER, True)
    analyze_metric_reaction_times.__wrapped__ = original
    module.analyze_metric_reaction_times = analyze_metric_reaction_times


__all__ = ["install"]
