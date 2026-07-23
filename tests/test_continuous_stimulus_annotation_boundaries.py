from __future__ import annotations

import pandas as pd

from neureptrace import continuous_stimulus_scan


def test_adjacent_scan_slices_do_not_duplicate_boundary_annotations() -> None:
    segments = [
        continuous_stimulus_scan.ScanSegment("slice0", 0.0, 1.0, 0.0),
        continuous_stimulus_scan.ScanSegment("slice1", 1.0, 2.0, 1.0),
    ]
    scan_events = pd.DataFrame(
        {
            "onset": [0.0, 1.0, 2.0],
            "stimulus_class": ["target", "target", "target"],
        }
    )

    annotations = continuous_stimulus_scan._annotation_table(
        scan_events=scan_events,
        segments=segments,
        onset_column="onset",
        label_column="stimulus_class",
        target_classes=["target"],
        annotation_latency=0.0,
        detection_window=None,
    )

    assert annotations.groupby("stream_id")["stimulus_onset_time"].apply(list).to_dict() == {
        "slice0": [0.0],
        "slice1": [0.0, 1.0],
    }


def test_overlapping_scan_slices_keep_independent_annotations() -> None:
    segments = [
        continuous_stimulus_scan.ScanSegment("slice0", 0.0, 2.0, 0.0),
        continuous_stimulus_scan.ScanSegment("slice1", 1.0, 3.0, 1.0),
    ]
    scan_events = pd.DataFrame({"onset": [1.5], "stimulus_class": ["target"]})

    annotations = continuous_stimulus_scan._annotation_table(
        scan_events=scan_events,
        segments=segments,
        onset_column="onset",
        label_column="stimulus_class",
        target_classes=["target"],
        annotation_latency=0.0,
        detection_window=None,
    )

    assert annotations[["stream_id", "stimulus_onset_time"]].to_dict("records") == [
        {"stream_id": "slice0", "stimulus_onset_time": 1.5},
        {"stream_id": "slice1", "stimulus_onset_time": 0.5},
    ]
