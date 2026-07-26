"""Normalize single reaction-time metric names before association analysis."""

from __future__ import annotations

import importlib
from functools import wraps

_PATCH_MARKER = "_neureptrace_reaction_time_metric_sequence_patch_installed"


def install() -> None:
    """Treat a bare metric-name string as one metric rather than characters."""

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
        normalized_metrics = (metrics,) if isinstance(metrics, str) else metrics
        return original(
            rows,
            normalized_metrics,
            reaction_time_column=reaction_time_column,
            participant_column=participant_column,
            min_trials=min_trials,
            include_pooled_within_participant=include_pooled_within_participant,
        )

    setattr(analyze_metric_reaction_times, _PATCH_MARKER, True)
    analyze_metric_reaction_times.__wrapped__ = original
    module.analyze_metric_reaction_times = analyze_metric_reaction_times


__all__ = ["install"]
