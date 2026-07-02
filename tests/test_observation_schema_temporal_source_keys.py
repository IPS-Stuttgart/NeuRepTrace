from __future__ import annotations

import pandas as pd

from neureptrace.observation_schema import validate_probability_observations


def test_temporal_profile_keeps_reused_sequence_ids_separate_by_source_file() -> None:
    frame = pd.DataFrame(
        {
            "source_path": ["run_a/observations.csv", "run_b/observations.csv"],
            "source_file": ["observations.csv", "observations.csv"],
            "subject": ["sub-01", "sub-01"],
            "sequence_id": ["trial-001", "trial-001"],
            "time": [0.1, 0.1],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "prob_class_0": [0.7, 0.2],
            "prob_class_1": [0.3, 0.8],
        }
    )

    report = validate_probability_observations(frame, profile="temporal-model")

    assert any(issue.code == "no_multi_point_sequence" for issue in report.errors)
    assert not any(issue.code == "duplicate_sequence_time" for issue in report.warnings)


def test_temporal_profile_keeps_reused_sample_indices_separate_by_session_run() -> None:
    frame = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "session": ["ses-01", "ses-02"],
            "run": ["run-01", "run-01"],
            "sample_index": [5, 5],
            "time": [0.0, 0.0],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "prob_class_0": [0.6, 0.4],
            "prob_class_1": [0.4, 0.6],
        }
    )

    report = validate_probability_observations(frame, profile="temporal-model")

    assert any(issue.code == "no_multi_point_sequence" for issue in report.errors)
    assert not any(issue.code == "duplicate_sequence_time" for issue in report.warnings)
