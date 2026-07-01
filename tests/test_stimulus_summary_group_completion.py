from __future__ import annotations

import pandas as pd

from neureptrace.stimulus_detection import summarize_stimulus_events


def test_grouped_stimulus_summary_keeps_zero_hit_groups_with_partial_events() -> None:
    events = pd.DataFrame(
        [
            {
                "subject": "subject1",
                "stream_id": "run1",
                "onset_time": 0.1,
                "stimulus_class": "target",
                "is_true_positive": True,
            }
        ]
    )
    annotations = pd.DataFrame(
        [
            {"subject": "subject1", "stream_id": "run1", "annotation_id": 1, "onset_time": 0.1, "stimulus_class": "target"},
            {"subject": "subject2", "stream_id": "run1", "annotation_id": 1, "onset_time": 0.1, "stimulus_class": "target"},
        ]
    )
    observations = pd.DataFrame(
        {
            "subject": ["subject1", "subject1", "subject2", "subject2"],
            "stream_id": ["run1", "run1", "run1", "run1"],
            "time": [0.0, 1.0, 0.0, 1.0],
        }
    )

    summary = summarize_stimulus_events(
        events,
        annotations=annotations,
        observations=observations,
        group_columns=("subject",),
    ).sort_values("subject", ignore_index=True)

    assert summary["subject"].tolist() == ["subject1", "subject2"]
    assert summary["n_detections"].tolist() == [1, 0]
    assert summary["false_negatives"].tolist() == [0, 1]
