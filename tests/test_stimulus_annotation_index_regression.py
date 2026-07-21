from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.event_detection import match_stimulus_annotations as match_event_annotations
from neureptrace.stimulus_detection import match_stimulus_annotations as match_stimulus_annotations


@pytest.mark.parametrize("matcher", [match_stimulus_annotations, match_event_annotations])
def test_annotation_matching_keeps_duplicate_event_indices_independent(matcher) -> None:
    event_index = pd.Index([7, 7], name="source_row")
    events = pd.DataFrame(
        [
            {"stream_id": "run-1", "onset_time": 0.1, "stimulus_class": "target", "stimulus_label": "target"},
            {"stream_id": "run-1", "onset_time": 1.1, "stimulus_class": "target", "stimulus_label": "target"},
        ],
        index=event_index,
    )
    annotations = pd.DataFrame(
        [
            {"stream_id": "run-1", "annotation_id": 101, "onset_time": 0.1, "stimulus_class": "target"},
            {"stream_id": "run-1", "annotation_id": 102, "onset_time": 1.1, "stimulus_class": "target"},
        ]
    )

    matched = matcher(events, annotations, stream_columns=("stream_id",), match_tolerance=0.01)

    assert matched.index.equals(event_index)
    assert matched["candidate_annotation_id"].tolist() == [101, 102]
    assert matched["matched_annotation_id"].tolist() == [101, 102]
    assert matched["is_true_positive"].tolist() == [True, True]
    assert matched["is_duplicate_detection"].tolist() == [False, False]
