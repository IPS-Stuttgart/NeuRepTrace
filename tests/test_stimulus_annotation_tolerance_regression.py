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
                "onset_time": 0.0,
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
                "onset_time": 100.0,
                "stimulus_class": "target",
            }
        ]
    )
    return events, annotations


@pytest.mark.parametrize("matcher", [match_stimulus_annotations, match_event_annotations])
@pytest.mark.parametrize("match_tolerance", [np.nan, np.inf, -np.inf, True, "not-a-number"])
def test_annotation_matching_rejects_invalid_match_tolerances(matcher, match_tolerance) -> None:
    events, annotations = _annotation_frames()

    with pytest.raises(ValueError, match="match_tolerance must be a non-negative finite number"):
        matcher(
            events,
            annotations,
            stream_columns=("stream_id",),
            match_tolerance=match_tolerance,
        )
