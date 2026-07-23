from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neureptrace import continuous_stimulus_scan


@pytest.mark.parametrize("slice_count", [0, -1, True, 1.5, np.nan, np.inf])
def test_build_scan_segments_rejects_invalid_slice_counts(slice_count: object) -> None:
    with pytest.raises(ValueError, match="slice_count must be a positive integer"):
        continuous_stimulus_scan.build_scan_segments(
            scan_raw=Path("scan_raw.fif"),
            scan_start=0.0,
            scan_stop=4.0,
            slice_duration=1.0,
            slice_count=slice_count,
        )


@pytest.mark.parametrize("slice_duration", [0.0, -1.0, True, np.nan, np.inf])
def test_build_scan_segments_rejects_invalid_slice_durations(slice_duration: object) -> None:
    with pytest.raises(ValueError, match="slice_duration must be a positive finite number"):
        continuous_stimulus_scan.build_scan_segments(
            scan_raw=Path("scan_raw.fif"),
            scan_start=0.0,
            scan_stop=4.0,
            slice_duration=slice_duration,
        )


def test_build_scan_segments_requires_duration_for_slice_selection() -> None:
    with pytest.raises(ValueError, match="slice_duration must be provided"):
        continuous_stimulus_scan.build_scan_segments(
            scan_raw=Path("scan_raw.fif"),
            scan_start=0.0,
            scan_stop=4.0,
            slice_count=2,
        )


def test_build_scan_segments_rejects_conflicting_slice_selectors() -> None:
    with pytest.raises(ValueError, match="slice_starts and slice_count are mutually exclusive"):
        continuous_stimulus_scan.build_scan_segments(
            scan_raw=Path("scan_raw.fif"),
            scan_start=0.0,
            scan_stop=4.0,
            slice_duration=1.0,
            slice_starts=[0.0],
            slice_count=1,
        )


@pytest.mark.parametrize("slice_starts", [[], [np.nan], [np.inf], [True]])
def test_build_scan_segments_rejects_invalid_explicit_starts(slice_starts: object) -> None:
    with pytest.raises(ValueError, match="slice_starts must contain"):
        continuous_stimulus_scan.build_scan_segments(
            scan_raw=Path("scan_raw.fif"),
            scan_start=0.0,
            scan_stop=4.0,
            slice_duration=1.0,
            slice_starts=slice_starts,
        )


def test_build_scan_segments_accepts_numpy_slice_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = SimpleNamespace(times=np.linspace(0.0, 4.0, 401))
    monkeypatch.setattr(continuous_stimulus_scan.mne.io, "read_raw_fif", lambda *args, **kwargs: raw)

    segments = continuous_stimulus_scan.build_scan_segments(
        scan_raw=Path("scan_raw.fif"),
        scan_start=0.0,
        scan_stop=4.0,
        slice_duration=1.0,
        slice_starts=np.array([0.0, 2.0]),
    )

    assert [(segment.start, segment.stop) for segment in segments] == [(0.0, 1.0), (2.0, 3.0)]
