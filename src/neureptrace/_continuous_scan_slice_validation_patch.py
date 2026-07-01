"""Validate continuous scan slice options before segment generation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_continuous_scan_slice_validation_patch_installed"


def _coerce_slice_starts(slice_starts: Sequence[float] | np.ndarray) -> list[float]:
    if isinstance(slice_starts, np.ndarray):
        if slice_starts.ndim == 0:
            values = [slice_starts.item()]
        else:
            values = slice_starts.ravel().tolist()
    else:
        values = list(slice_starts)
    if not values:
        raise ValueError("slice_starts must contain at least one start when provided.")

    starts: list[float] = []
    for value in values:
        start = float(value)
        if not np.isfinite(start):
            raise ValueError("slice_starts must contain finite values.")
        starts.append(start)
    return starts


def _positive_slice_count(slice_count: int | np.integer[Any]) -> int:
    if isinstance(slice_count, (bool, np.bool_)) or not isinstance(slice_count, (int, np.integer)):
        raise ValueError("slice_count must be a positive integer.")
    parsed = int(slice_count)
    if parsed < 1:
        raise ValueError("slice_count must be a positive integer.")
    return parsed


def install() -> None:
    import neureptrace.continuous_stimulus_scan as continuous_stimulus_scan

    original = continuous_stimulus_scan.build_scan_segments
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def build_scan_segments(
        *,
        scan_raw: Path,
        scan_start: float | None,
        scan_stop: float | None,
        slice_duration: float | None = None,
        slice_starts: Sequence[float] | np.ndarray | None = None,
        slice_count: int | np.integer[Any] | None = None,
        slice_seed: int = 13,
        scan_events: pd.DataFrame | None = None,
        onset_column: str = "onset",
        label_column: str = "stimulus_class",
        target_classes: Sequence[str] | None = None,
        threshold_window: tuple[float, float] | None = None,
        detection_window: tuple[float, float] | None = None,
        require_target_event: bool = False,
        exclude_events_from_threshold_window: bool = False,
        stream_id: str | None = None,
    ) -> list[continuous_stimulus_scan.ScanSegment]:
        raw = continuous_stimulus_scan.mne.io.read_raw_fif(scan_raw, preload=False, verbose="error")
        raw_start = 0.0 if scan_start is None else float(scan_start)
        raw_stop = float(raw.times[-1]) if scan_stop is None else float(scan_stop)
        if raw_stop <= raw_start:
            raise ValueError("scan_stop must be greater than scan_start.")

        base_stream_id = stream_id or continuous_stimulus_scan._safe_stream_id(scan_raw)
        if slice_duration is None:
            if slice_starts is not None:
                raise ValueError("slice_starts requires slice_duration.")
            if slice_count is not None:
                raise ValueError("slice_count requires slice_duration.")
            return [continuous_stimulus_scan.ScanSegment(base_stream_id, raw_start, raw_stop, 0.0)]

        slice_duration = float(slice_duration)
        if slice_duration <= 0.0 or not np.isfinite(slice_duration):
            raise ValueError("slice_duration must be positive and finite.")
        if slice_duration > raw_stop - raw_start:
            raise ValueError("slice_duration must fit within the scan interval.")

        starts: list[float]
        if slice_starts is not None:
            starts = _coerce_slice_starts(slice_starts)
        elif slice_count is not None:
            requested_count = _positive_slice_count(slice_count)
            rng = np.random.default_rng(slice_seed)
            starts = []
            target_set = set(map(str, target_classes or [])) or None
            tries = 0
            while len(starts) < requested_count and tries < max(1000, requested_count * 500):
                tries += 1
                start = float(rng.uniform(raw_start, raw_stop - slice_duration))
                if scan_events is not None and exclude_events_from_threshold_window and threshold_window is not None:
                    if continuous_stimulus_scan._event_mask_in_window(
                        scan_events,
                        onset_column=onset_column,
                        start=start + threshold_window[0],
                        stop=start + threshold_window[1],
                        label_column=label_column,
                    ).any():
                        continue
                if scan_events is not None and require_target_event and detection_window is not None:
                    if not continuous_stimulus_scan._event_mask_in_window(
                        scan_events,
                        onset_column=onset_column,
                        start=start + detection_window[0],
                        stop=start + detection_window[1],
                        labels=target_set,
                        label_column=label_column,
                    ).any():
                        continue
                starts.append(start)
            if len(starts) < requested_count:
                raise ValueError(f"Only selected {len(starts)} random slice(s); requested {requested_count}.")
        else:
            starts = list(np.arange(raw_start, raw_stop - slice_duration + 1e-12, slice_duration))

        segments = []
        for index, start in enumerate(starts):
            stop = start + slice_duration
            if start < raw_start or stop > raw_stop:
                raise ValueError(f"Slice [{start}, {stop}] is outside scan interval [{raw_start}, {raw_stop}].")
            segments.append(continuous_stimulus_scan.ScanSegment(f"{base_stream_id}_slice{index:03d}", start, stop, start))
        return segments

    setattr(build_scan_segments, _PATCH_MARKER, True)
    continuous_stimulus_scan.build_scan_segments = build_scan_segments


__all__ = ["install"]
