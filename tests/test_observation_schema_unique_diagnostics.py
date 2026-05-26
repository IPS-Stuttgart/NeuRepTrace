from __future__ import annotations

import pandas as pd

from neureptrace.observation_schema import validate_probability_observations


def _base_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0],
            "prob_class_0": [0.4],
            "prob_class_1": [0.6],
        }
    )


def test_probability_above_one_reported_once() -> None:
    frame = _base_observations()
    frame.loc[0, "prob_class_0"] = 1.2
    frame.loc[0, "prob_class_1"] = 0.0

    report = validate_probability_observations(frame)

    matches = [
        issue
        for issue in report.errors
        if issue.code == "probability_above_one" and issue.column == "prob_class_0"
    ]
    assert len(matches) == 1


def test_non_finite_probability_reported_once() -> None:
    frame = _base_observations()
    frame.loc[0, "prob_class_0"] = 1e309
    frame.loc[0, "prob_class_1"] = 0.0

    report = validate_probability_observations(frame)

    matches = [
        issue
        for issue in report.errors
        if issue.code == "non_finite_probability" and issue.column == "prob_class_0"
    ]
    assert len(matches) == 1
