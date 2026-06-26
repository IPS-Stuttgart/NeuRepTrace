from __future__ import annotations

import pandas as pd

from neureptrace.observation_schema import validate_probability_observations


def _observation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "stream_id": ["stream-001", "stream-001"],
            "time": [0.0, 0.1],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "prob_class_0": [0.7, 0.2],
            "prob_class_1": [0.3, 0.8],
        }
    )


def test_validate_observations_accepts_single_group_column_string() -> None:
    report = validate_probability_observations(_observation_frame(), group_columns="subject")

    assert report.is_valid
    assert not any(issue.code == "missing_group_column" for issue in report.errors)


def test_validate_stimulus_profile_accepts_single_stream_column_string() -> None:
    report = validate_probability_observations(
        _observation_frame(),
        profile="stimulus-detection",
        stream_columns="stream_id",
    )

    assert report.is_valid
    assert not any(issue.code == "missing_stream_column" for issue in report.errors)
