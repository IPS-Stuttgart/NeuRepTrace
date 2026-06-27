import numpy as np
import pandas as pd

from neureptrace.onset_detection import summarize_onset_events, summarize_threshold_crossings


def test_summarize_onset_events_parses_string_boolean_columns():
    events = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01", "sub-01"],
            "detected": ["True", "False", "True"],
            "detected_before_zero": ["False", "False", "True"],
            "is_correct_at_detection": ["True", "False", "False"],
            "detection_latency": [0.10, np.nan, -0.05],
            "detection_run_duration": [0.02, np.nan, 0.01],
            "detection_run_length": [2, 0, 1],
        }
    )

    summary = summarize_onset_events(events)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["detected_count"] == 2
    assert row["false_alarm_count"] == 1
    assert row["post_zero_detected_count"] == 1
    assert row["correct_detection_count"] == 1
    assert row["post_detection_latency_median"] == 0.10


def test_summarize_threshold_crossings_parses_string_boolean_columns():
    thresholded = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01", "sub-01", "sub-01"],
            "sequence_id": [0, 0, 0, 0],
            "time": [-0.20, -0.10, 0.10, 0.20],
            "above_threshold": ["False", "False", "True", "False"],
            "is_correct": ["False", "False", "True", "False"],
            "score_threshold": [0.5, 0.5, 0.5, 0.5],
            "score_column": ["confidence", "confidence", "confidence", "confidence"],
            "threshold_method": ["point", "point", "point", "point"],
            "threshold_quantile": [0.95, 0.95, 0.95, 0.95],
        }
    )

    summary = summarize_threshold_crossings(
        thresholded,
        baseline_window=(-0.20, -0.10),
        detection_window=(0.0, 0.30),
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["baseline_false_positive_count"] == 0
    assert row["baseline_false_positive_sequence_count"] == 0
    assert row["post_stimulus_detection_count"] == 1
    assert row["post_stimulus_detection_sequence_count"] == 1
    assert row["post_stimulus_correct_detection_count"] == 1
