from __future__ import annotations

import pandas as pd

from neureptrace.observation_schema import validate_probability_observations


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "time": [0.0],
        "decoder": ["logistic"],
        "backend": ["sklearn"],
        "emission_mode": ["calibrated"],
        "split_id": ["split-001"],
        "seed": [13],
        "train_time": [0.0],
        "test_time": [0.0],
        "preprocessing_hash": ["pre-123"],
        "model_hash": ["model-456"],
        "true_label": [1],
        "predicted_label": [1],
        "probability_true_class": [1.0],
        "confidence": [1.0],
        "predicted_class": ["face"],
        "prob_class_0": [0.0],
        "prob_class_1": [1.0],
        "class_0": ["object"],
        "class_1": ["face"],
    })


def test_integer_like_float_labels_are_valid() -> None:
    frame = _frame()
    frame["true_label"] = [1.0]
    frame["predicted_label"] = [1.0]

    report = validate_probability_observations(frame, profile="canonical")

    assert report.is_valid


def test_fractional_true_label_is_invalid() -> None:
    frame = _frame()
    frame.loc[0, "true_label"] = 0.5

    report = validate_probability_observations(frame, profile="canonical")

    assert any(issue.code == "invalid_true_label" for issue in report.errors)


def test_fractional_predicted_label_is_invalid() -> None:
    frame = _frame()
    frame.loc[0, "predicted_label"] = 0.5

    report = validate_probability_observations(frame, profile="canonical")

    assert any(issue.code == "invalid_predicted_label" for issue in report.errors)
