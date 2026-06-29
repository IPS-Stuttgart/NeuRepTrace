from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.observation_schema import validate_probability_observations


def _valid_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "sequence_id": ["trial-001", "trial-001"],
            "time": [-0.1, -0.08],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "prob_class_0": [0.6, 0.45],
            "prob_class_1": [0.4, 0.55],
        }
    )


def _valid_canonical_observations() -> pd.DataFrame:
    frame = _valid_observations()
    frame["session"] = ["run-01", "run-01"]
    frame["fold"] = [0, 0]
    frame["split_id"] = ["split-001", "split-001"]
    frame["seed"] = [13, 13]
    frame["backend"] = ["sklearn", "sklearn"]
    frame["train_time"] = frame["time"]
    frame["test_time"] = frame["time"]
    frame["true_label"] = [1, 1]
    frame["predicted_label"] = [0, 1]
    frame["true_class"] = ["face", "face"]
    frame["predicted_class"] = ["object", "face"]
    frame["probability_true_class"] = [0.4, 0.55]
    frame["confidence"] = [0.6, 0.55]
    frame["class_0"] = ["object", "object"]
    frame["class_1"] = ["face", "face"]
    frame["calibration_fold"] = ["", ""]
    frame["preprocessing_hash"] = ["pre-123", "pre-123"]
    frame["model_hash"] = ["model-456", "model-456"]
    return frame


def _put_bool(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    mutated = frame.copy()
    mutated[column] = mutated[column].astype(object)
    mutated.loc[0, column] = True
    return mutated


@pytest.mark.parametrize("column", ["time", "prob_class_0", "prob_class_1"])
def test_generic_observation_numeric_fields_reject_booleans(column: str) -> None:
    report = validate_probability_observations(_put_bool(_valid_observations(), column))

    assert not report.is_valid
    assert any(
        issue.code == "boolean_numeric_value" and issue.column == column and issue.row == 0
        for issue in report.errors
    )


@pytest.mark.parametrize(
    "column",
    ["seed", "train_time", "test_time", "confidence", "probability_true_class"],
)
def test_canonical_observation_numeric_fields_reject_booleans(column: str) -> None:
    report = validate_probability_observations(_put_bool(_valid_canonical_observations(), column), profile="canonical")

    assert not report.is_valid
    assert any(
        issue.code == "boolean_numeric_value" and issue.column == column and issue.row == 0
        for issue in report.errors
    )


@pytest.mark.parametrize("column", ["true_label", "predicted_label"])
def test_canonical_label_booleans_are_not_accepted_as_integer_labels(column: str) -> None:
    report = validate_probability_observations(_put_bool(_valid_canonical_observations(), column), profile="canonical")

    assert not report.is_valid
    assert any(
        issue.code == "boolean_numeric_value" and issue.column == column and issue.row == 0
        for issue in report.errors
    )
