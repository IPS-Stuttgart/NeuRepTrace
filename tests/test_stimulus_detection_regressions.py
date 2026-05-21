from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.stimulus_detection import match_stimulus_annotations, summarize_stimulus_events
from neureptrace.streaming_stimulus_detection import StimulusDetectionConfig, StreamingStimulusDetector


def test_annotation_ids_are_scoped_by_stream_when_matching() -> None:
    events = pd.DataFrame(
        [
            {"stream_id": "run1", "onset_time": 0.1, "stimulus_class": "target", "stimulus_label": "target"},
            {"stream_id": "run2", "onset_time": 0.1, "stimulus_class": "target", "stimulus_label": "target"},
        ]
    )
    annotations = pd.DataFrame(
        [
            {"stream_id": "run1", "annotation_id": 1, "onset_time": 0.1, "stimulus_class": "target"},
            {"stream_id": "run2", "annotation_id": 1, "onset_time": 0.1, "stimulus_class": "target"},
        ]
    )

    matched = match_stimulus_annotations(events, annotations, stream_columns=("stream_id",), match_tolerance=0.01)

    assert matched["is_true_positive"].tolist() == [True, True]
    assert matched["is_duplicate_detection"].tolist() == [False, False]


def test_false_alarm_rate_uses_stream_fallback_duration() -> None:
    events = pd.DataFrame(
        [
            {
                "subject": "subject1",
                "stream_id": "run1",
                "onset_time": 0.5,
                "stimulus_class": "target",
                "is_true_positive": False,
            }
        ]
    )
    observations = pd.DataFrame(
        {
            "subject": ["subject1", "subject1", "subject1", "subject1"],
            "stream_id": ["run1", "run1", "run2", "run2"],
            "time": [0.0, 1.0, 0.0, 1.0],
        }
    )

    summary = summarize_stimulus_events(events, observations=observations, group_columns=("subject",))

    assert np.isclose(summary.loc[0, "false_alarms_per_minute"], 30.0)


def test_empty_grouped_stimulus_summary_reports_zero_detections() -> None:
    events = pd.DataFrame(columns=["subject", "onset_time", "stimulus_class"])
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
    )

    assert summary["subject"].tolist() == ["subject1", "subject2"]
    assert summary["n_detections"].tolist() == [0, 0]
    assert summary["false_negatives"].tolist() == [1, 1]


def test_streaming_event_indices_restart_for_each_group_stream_partition() -> None:
    thresholds = pd.DataFrame(
        [
            {
                "subject": subject,
                "stimulus_label": 0,
                "stimulus_class": "target",
                "score_column": "prob_class_0",
                "score_mode": "class_probability",
                "score_threshold": 0.5,
                "threshold_method": "point",
                "threshold_quantile": 0.95,
                "threshold_window_start": -0.2,
                "threshold_window_stop": -0.1,
            }
            for subject in ("subject1", "subject2")
        ]
    )
    detector = StreamingStimulusDetector(
        StimulusDetectionConfig(group_columns=("subject",), stream_columns=("stream_id",)),
        thresholds,
    )

    events: list[dict[str, object]] = []
    for subject in ("subject1", "subject2"):
        events.extend(
            detector.update(
                {
                    "subject": subject,
                    "stream_id": "shared-run-id",
                    "time": 0.0,
                    "prob_class_0": 0.9,
                    "predicted_label": 0,
                    "predicted_class": "target",
                }
            )
        )
        events.extend(
            detector.update(
                {
                    "subject": subject,
                    "stream_id": "shared-run-id",
                    "time": 1.0,
                    "prob_class_0": 0.1,
                    "predicted_label": 0,
                    "predicted_class": "target",
                }
            )
        )

    assert [(event["subject"], event["event_index"]) for event in events] == [("subject1", 0), ("subject2", 0)]
