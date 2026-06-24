"""Behavioral-data utilities for NeuRepTrace analyses."""

from __future__ import annotations

from neureptrace import _reaction_time_trial_index_base_patch
from neureptrace.behavior.reaction_time import (
    REACTION_TIME_FIELD_CANDIDATES,
    ReactionTimeCsvConfig,
    ReactionTimeUnavailableError,
    analyze_metric_reaction_times,
    extract_reaction_times_from_metadata,
    join_reaction_times,
    load_reaction_time_csv,
    reaction_time_rows_from_values,
)

_reaction_time_trial_index_base_patch.install()

__all__ = [
    "REACTION_TIME_FIELD_CANDIDATES",
    "ReactionTimeCsvConfig",
    "ReactionTimeUnavailableError",
    "analyze_metric_reaction_times",
    "extract_reaction_times_from_metadata",
    "join_reaction_times",
    "load_reaction_time_csv",
    "reaction_time_rows_from_values",
]
