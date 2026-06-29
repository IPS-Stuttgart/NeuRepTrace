import numpy as np
import pandas as pd

from neureptrace.onset_detection import (
    annotate_threshold_crossings,
    detect_onsets,
    summarize_onset_events,
    summarize_threshold_crossings,
)


def _observation_frame() -> pd.DataFrame:
    rows = []
    traces = {
        0: [(-0.20, 0.55), (-0.10, 0.58), (0.05, 0.62), (0.15, 0.92), (0.25, 0.88)],
        1: [(-0.20, 0.57), (-0.10, 0.59), (0.05, 0.90), (0.15, 0.86), (0.25, 0.84)],
        2: [(-0.20, 0.56), (-0.10, 0.91), (0.05, 0.85), (0.15, 0.80), (0.25, 0.77)],
        3: [(-0.20, 0.53), (-0.10, 0.54), (0.05, 0.55), (0.15, 0.56), (0.25, 0.57)],
    }
    for sequence_id, trace in traces.items():
        true_label = sequence_id % 2
        for time, confidence in trace:
            predicted_label = true_label if confidence >= 0.80 else 1 - true_label
            probabilities = np.array([0.0, 0.0])
            probabilities[predicted_label] = confidence
            probabilities[1 - predicted_label] = 1.0 - confidence
            rows.append(
                {
                    "subject": "sub-01",
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": time,
                    "sample_index": sequence_id,
                    "sequence_id": sequence_id,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "prob_class_0": probabilities[0],
                    "prob_class_1": probabilities[1],
                }
            )
    return pd.DataFrame(rows)


def test_onset_helpers_preserve_missing_group_metadata() -> None:
    observations = _observation_frame()
    missing_sequence = observations["sequence_id"] == 3
    observations.loc[missing_sequence, "emission_mode"] = np.nan

    thresholded = annotate_threshold_crossings(
        observations,
        threshold_window=(-0.20, -0.10),
        threshold_quantile=0.875,
    )

    assert len(thresholded) == len(observations)
    assert thresholded.loc[thresholded["sequence_id"].eq(3), "emission_mode"].isna().all()

    events = detect_onsets(
        thresholded,
        threshold_window=(-0.20, -0.10),
        threshold_quantile=0.875,
    )

    events_by_sequence = events.set_index("sequence_id")
    assert len(events_by_sequence) == 4
    assert pd.isna(events_by_sequence.loc[3, "emission_mode"])

    event_summary = summarize_onset_events(events)
    assert event_summary["n_sequences"].sum() == 4
    assert event_summary["emission_mode"].isna().any()

    threshold_summary = summarize_threshold_crossings(
        thresholded,
        baseline_window=(-0.20, -0.10),
        detection_window=(0.0, float("inf")),
    )
    assert threshold_summary["baseline_n_observations"].sum() == 8
    assert threshold_summary["post_stimulus_n_observations"].sum() == 12
    assert threshold_summary["emission_mode"].isna().any()
