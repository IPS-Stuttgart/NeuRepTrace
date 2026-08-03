from pathlib import Path

import numpy as np
import pytest

from neureptrace.continuous_stimulus_scan import build_scan_segments


@pytest.mark.parametrize("slice_count", [0, -1, True, np.bool_(False), 1.5, "2"])
def test_build_scan_segments_rejects_invalid_slice_count_before_file_access(slice_count) -> None:
    with pytest.raises(ValueError, match="positive non-boolean integer"):
        build_scan_segments(
            scan_raw=Path("does-not-exist_raw.fif"),
            scan_start=None,
            scan_stop=None,
            slice_duration=1.0,
            slice_count=slice_count,
        )


def test_build_scan_segments_accepts_numpy_integer_slice_count(monkeypatch) -> None:
    class _Raw:
        times = np.array([0.0, 10.0])

    monkeypatch.setattr(
        "neureptrace.continuous_stimulus_scan.mne.io.read_raw_fif",
        lambda *args, **kwargs: _Raw(),
    )

    segments = build_scan_segments(
        scan_raw=Path("unused_raw.fif"),
        scan_start=0.0,
        scan_stop=10.0,
        slice_duration=1.0,
        slice_count=np.int64(2),
        slice_seed=7,
    )

    assert len(segments) == 2
    assert all(segment.stop - segment.start == pytest.approx(1.0) for segment in segments)
