from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.stimulus_detection import summarize_stimulus_events


def test_summarize_stimulus_events_parses_string_boolean_columns_after_csv_roundtrip():
    events = pd.DataFrame(
        {
            "stream_id": ["run-1", "run-1", "run-1"],
            "stimulus_class": ["A", "A", "B"],
            "matched_annotation_class": ["A", "", ""],
            "is_true_positive": ["True", "False", "False"],
            "is_duplicate_detection": ["False", "True", "False"],
            "latency": [0.01, np.nan, np.nan],
        }
    )
    annotations = pd.DataFrame(
        [
            {"stream_id": "run-1", "annotation_id": 1, "stimulus_class": "A", "onset_time": 0.0},
        ]
    )

    summary = summarize_stimulus_events(events, annotations=annotations)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_detections"] == 3
    assert row["n_annotations"] == 1
    assert row["true_positives"] == 1
    assert row["false_positives"] == 2
    assert row["false_negatives"] == 0
    assert row["duplicate_detections"] == 1
    assert row["precision"] == 1 / 3
    assert row["recall"] == 1.0
    assert row["class_accuracy_for_matched_events"] == 1.0
    assert row["latency_mean"] == 0.01


def test_summarize_stimulus_events_rejects_non_boolean_summary_values():
    events = pd.DataFrame(
        {
            "stream_id": ["run-1"],
            "stimulus_class": ["A"],
            "is_true_positive": ["maybe"],
        }
    )

    with pytest.raises(ValueError, match="is_true_positive"):
        summarize_stimulus_events(events)
