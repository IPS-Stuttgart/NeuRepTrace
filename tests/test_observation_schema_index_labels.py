from __future__ import annotations

import pandas as pd

from neureptrace.observation_schema import validate_probability_observations


def _observations_with_string_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "sequence_id": ["trial-001", "trial-001"],
            "time": [0.0, 0.1],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "prob_class_0": [1.2, 0.4],
            "prob_class_1": [0.0, 0.6],
        },
        index=["sample-a", "sample-b"],
    )


def test_validation_reports_positions_for_non_integer_index_labels() -> None:
    frame = _observations_with_string_index()

    report = validate_probability_observations(frame)

    issue = next(issue for issue in report.errors if issue.code == "probability_above_one")
    assert issue.row == 0
    assert frame.index.tolist() == ["sample-a", "sample-b"]


def test_temporal_identifier_diagnostic_uses_position_with_string_index() -> None:
    frame = _observations_with_string_index()
    frame.loc["sample-a", "prob_class_0"] = 0.6
    frame.loc["sample-a", "prob_class_1"] = 0.4
    frame.loc["sample-a", "sequence_id"] = ""

    report = validate_probability_observations(frame, profile="temporal-model")

    issue = next(issue for issue in report.errors if issue.code == "missing_sequence_identifier_value")
    assert issue.row == 0
    assert frame.index.tolist() == ["sample-a", "sample-b"]
