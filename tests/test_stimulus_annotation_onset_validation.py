from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.event_detection import match_stimulus_annotations as match_event_annotations
from neureptrace.stimulus_detection import match_stimulus_annotations as match_stimulus_annotations


def _annotation_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.DataFrame(
        [
            {
                "stream_id": "run-1",
                "onset_time": 1.0,
                "stimulus_class": "target",
                "stimulus_label": "target",
            }
        ]
    )
    annotations = pd.DataFrame(
        [
            {
                "stream_id": "run-1",
                "annotation_id": 101,
                "onset_time": 1.05,
                "stimulus_class": "target",
            }
        ]
    )
    events["onset_time"] = events["onset_time"].astype(object)
    annotations["onset_time"] = annotations["onset_time"].astype(object)
    return events, annotations


@pytest.mark.parametrize("matcher", [match_stimulus_annotations, match_event_annotations])
@pytest.mark.parametrize("frame_name", ["events", "annotations"])
@pytest.mark.parametrize("invalid_time", [np.nan, np.inf, -np.inf, True, 1.0 + 2.0j, "not-a-time", None])
def test_annotation_matching_rejects_invalid_onset_times(matcher, frame_name, invalid_time) -> None:
    events, annotations = _annotation_frames()
    frame = events if frame_name == "events" else annotations
    frame.loc[0, "onset_time"] = invalid_time

    with pytest.raises(ValueError, match=rf"{frame_name} onset_time must contain only finite real numbers"):
        matcher(
            events,
            annotations,
            stream_columns=("stream_id",),
            match_tolerance=0.1,
        )


@pytest.mark.parametrize("matcher", [match_stimulus_annotations, match_event_annotations])
def test_annotation_matching_accepts_numeric_string_onset_times(matcher) -> None:
    events, annotations = _annotation_frames()
    events.loc[0, "onset_time"] = "1.0"
    annotations.loc[0, "onset_time"] = "1.05"

    matched = matcher(
        events,
        annotations,
        stream_columns=("stream_id",),
        match_tolerance=0.1,
    )

    assert bool(matched.loc[0, "is_true_positive"])
    assert matched.loc[0, "matched_annotation_id"] == 101
