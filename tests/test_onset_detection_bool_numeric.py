from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace import onset_detection


def test_public_onset_probability_true_class_ignores_boolean_labels() -> None:
    frame = pd.DataFrame(
        {
            "true_label": [True, np.bool_(False), "1.0"],
            "prob_class_0": [0.2, 0.8, 0.3],
            "prob_class_1": [0.8, 0.2, 0.7],
        }
    )

    scores = onset_detection._score_values(frame, "probability_true_class")

    assert np.isnan(scores.iloc[0])
    assert np.isnan(scores.iloc[1])
    assert scores.iloc[2] == 0.7


def test_public_onset_prediction_columns_ignore_boolean_labels() -> None:
    frame = pd.DataFrame(
        {
            "predicted_label": [True, np.bool_(False), "1.0"],
            "class_0": ["zero", "zero", "zero"],
            "class_1": ["one", "one", "one"],
            "prob_class_0": [0.9, 0.1, 0.9],
            "prob_class_1": [0.1, 0.9, 0.1],
        }
    )

    result = onset_detection._ensure_prediction_columns(frame)

    assert result["predicted_class"].tolist() == ["zero", "one", "one"]


def test_public_onset_confidence_rejects_boolean_values() -> None:
    frame = pd.DataFrame(
        {
            "confidence": [True, 0.7],
            "prob_class_0": [0.2, 0.4],
            "prob_class_1": [0.8, 0.6],
        }
    )

    with pytest.raises(ValueError, match="confidence values.*booleans"):
        onset_detection._score_values(frame, "confidence")


def test_public_onset_correctness_rejects_boolean_numeric_labels() -> None:
    assert not onset_detection._is_correct_detection(pd.Series({"true_label": True, "predicted_label": 1}))
    assert not onset_detection._is_correct_detection(pd.Series({"true_label": 1, "predicted_label": np.bool_(True)}))


def test_detect_onsets_does_not_count_boolean_label_as_correct_integer_label() -> None:
    observations = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "sequence_id": ["trial-1", "trial-1"],
            "time": [-0.1, 0.1],
            "confidence": [0.1, 0.9],
            "true_label": [True, True],
            "predicted_label": [1, 1],
            "prob_class_0": [0.2, 0.2],
            "prob_class_1": [0.8, 0.8],
        }
    )

    events = onset_detection.detect_onsets(
        observations,
        threshold_window=(-0.2, -0.05),
        threshold_quantile=1.0,
        detection_window=(0.0, 1.0),
        score_column="confidence",
    )

    assert bool(events.loc[0, "detected"])
    assert not bool(events.loc[0, "is_correct_at_detection"])
