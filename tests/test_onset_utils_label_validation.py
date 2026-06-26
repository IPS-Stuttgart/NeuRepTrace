import numpy as np
import pandas as pd
import pytest

from neureptrace._onset_utils import (
    ensure_prediction_columns,
    is_correct_detection,
    score_values,
)


def test_probability_true_class_scores_ignore_fractional_labels():
    frame = pd.DataFrame(
        {
            "true_label": [0, 0.5, "1.0", np.nan],
            "prob_class_0": [0.8, 0.9, 0.3, 0.4],
            "prob_class_1": [0.2, 0.1, 0.7, 0.6],
        }
    )

    scores = score_values(frame, "probability_true_class")

    assert scores.iloc[0] == 0.8
    assert np.isnan(scores.iloc[1])
    assert scores.iloc[2] == 0.7
    assert np.isnan(scores.iloc[3])


def test_probability_true_class_scores_ignore_boolean_labels():
    frame = pd.DataFrame(
        {
            "true_label": [True, np.bool_(False), "1.0"],
            "prob_class_0": [0.2, 0.8, 0.3],
            "prob_class_1": [0.8, 0.2, 0.7],
        }
    )

    scores = score_values(frame, "probability_true_class")

    assert np.isnan(scores.iloc[0])
    assert np.isnan(scores.iloc[1])
    assert scores.iloc[2] == 0.7


def test_confidence_scores_must_be_valid_probabilities():
    frame = pd.DataFrame({"confidence": [0.8, 1.2], "prob_class_0": [0.8, 0.2], "prob_class_1": [0.2, 0.8]})

    with pytest.raises(ValueError, match="confidence values must lie"):
        score_values(frame, "confidence")


def test_ensure_prediction_columns_does_not_truncate_fractional_labels():
    frame = pd.DataFrame(
        {
            "predicted_label": [0.5, "1.0"],
            "class_0": ["zero", "zero"],
            "class_1": ["one", "one"],
            "prob_class_0": [0.1, 0.9],
            "prob_class_1": [0.9, 0.1],
        }
    )

    result = ensure_prediction_columns(frame)

    assert result["predicted_class"].tolist() == ["one", "one"]


def test_ensure_prediction_columns_ignores_boolean_labels():
    frame = pd.DataFrame(
        {
            "predicted_label": [True, np.bool_(False), "1.0"],
            "class_0": ["zero", "zero", "zero"],
            "class_1": ["one", "one", "one"],
            "prob_class_0": [0.9, 0.1, 0.9],
            "prob_class_1": [0.1, 0.9, 0.1],
        }
    )

    result = ensure_prediction_columns(frame)

    assert result["predicted_class"].tolist() == ["zero", "one", "one"]


def test_is_correct_detection_rejects_fractional_numeric_labels():
    assert not is_correct_detection(pd.Series({"true_label": 0.5, "predicted_label": 0}))
    assert is_correct_detection(pd.Series({"true_label": "1.0", "predicted_label": 1}))


def test_is_correct_detection_rejects_boolean_labels():
    assert not is_correct_detection(pd.Series({"true_label": True, "predicted_label": 1}))
    assert not is_correct_detection(pd.Series({"true_label": 1, "predicted_label": np.bool_(True)}))
