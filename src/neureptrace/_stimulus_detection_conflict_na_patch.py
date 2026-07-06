"""Preserve stimulus events with missing conflict-resolution partition keys."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import pandas as pd

_PATCH_MARKER = "_neureptrace_stimulus_conflict_na_patch_installed"


def install() -> None:
    """Patch stimulus conflict resolution to keep NaN group keys."""

    public = importlib.import_module("neureptrace._stimulus_detection_public")
    original = getattr(public, "_resolve_event_conflicts", None)
    if original is None or getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _resolve_event_conflicts(events: pd.DataFrame, *, partition_columns: Sequence[str], conflict_resolution: str) -> pd.DataFrame:
        if conflict_resolution not in public.CONFLICT_RESOLUTION_MODES:
            raise ValueError(f"conflict_resolution must be one of {public.CONFLICT_RESOLUTION_MODES}.")
        if events.empty or conflict_resolution == "none":
            return events

        frames: list[pd.DataFrame] = []
        present_partition_columns = public._present_columns(events, partition_columns)
        grouped: Any = events.groupby(present_partition_columns, sort=True, dropna=False) if present_partition_columns else [((), events)]
        for _, partition_events in grouped:
            if conflict_resolution == "winner_take_all":
                frames.append(public._resolve_winner_take_all(partition_events))
            elif conflict_resolution == "non_max_suppression":
                frames.append(public._resolve_non_max_suppression(partition_events))
            elif conflict_resolution == "highest_peak_per_window":
                frames.append(public._resolve_highest_peak_per_window(partition_events))
        return pd.concat(frames, ignore_index=False) if frames else events.iloc[0:0].copy()

    setattr(_resolve_event_conflicts, _PATCH_MARKER, True)
    _resolve_event_conflicts.__wrapped__ = original
    public._resolve_event_conflicts = _resolve_event_conflicts


__all__ = ["install"]
