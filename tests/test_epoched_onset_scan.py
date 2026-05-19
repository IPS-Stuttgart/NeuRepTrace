import pandas as pd

from neureptrace.epoched_onset_scan import (
    epoched_prediction_traces_to_observations,
    run_epoched_onset_scan,
    standardize_epoched_onset_observations,
)


def test_epoched_prediction_traces_run_through_onset_detector():
    observations = epoched_prediction_traces_to_observations(
        scores=[
            [0.10, 0.20, 0.90, 0.80],
            [0.20, 0.30, 0.25, 0.70],
        ],
        predicted_labels=[
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        true_labels=[1, 1],
        sequence_ids=["trial-0", "trial-1"],
        times=[-0.2, -0.1, 0.1, 0.2],
        window_size=0.05,
        metadata={"subject": "sub-01", "decoder": "svm", "emission_mode": "score"},
    )

    result = run_epoched_onset_scan(
        observations,
        threshold_window=(-0.2, -0.1),
        threshold_quantile=1.0,
        detection_start=0.0,
    )

    events = result.events.set_index("sequence_id")
    assert result.observations["score_threshold"].notna().all()
    assert events.loc["trial-0", "detection_time"] == 0.1
    assert events.loc["trial-1", "detection_time"] == 0.2
    assert events.loc["trial-0", "is_correct_at_detection"]
    assert events.loc["trial-1", "is_correct_at_detection"]


def test_standardize_epoched_onset_observations_accepts_project_column_aliases():
    project_rows = pd.DataFrame(
        {
            "validation_trial_index": [0, 0, 1, 1],
            "scan_window_center_s": [-0.1, 0.1, -0.1, 0.1],
            "scan_window_start_s": [-0.15, 0.05, -0.15, 0.05],
            "scan_window_stop_s": [-0.05, 0.15, -0.05, 0.15],
            "true_label": [1, 1, 1, 1],
            "predicted_label": [0, 1, 0, 1],
            "stimulus_score": [0.1, 0.9, 0.2, 0.8],
        }
    )

    observations = standardize_epoched_onset_observations(
        project_rows,
        sequence_column="validation_trial_index",
        time_column="scan_window_center_s",
        window_start_column="scan_window_start_s",
        window_stop_column="scan_window_stop_s",
        correct_column=None,
        score_column="stimulus_score",
    )

    assert observations["sequence_id"].tolist() == [0, 0, 1, 1]
    assert observations["time"].tolist() == [-0.1, 0.1, -0.1, 0.1]
    assert observations["window_start"].tolist() == [-0.15, 0.05, -0.15, 0.05]
    assert observations["is_correct"].tolist() == [False, True, False, True]
