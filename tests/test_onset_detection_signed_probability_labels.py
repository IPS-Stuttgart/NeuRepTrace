from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.onset_detection import annotate_threshold_crossings, detect_onsets


def _signed_label_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "sequence_id": [0, 0],
            "time": [-0.2, 0.1],
            "true_label": [-1, -1],
            "true_class": ["negative", "negative"],
            "class_-1": ["negative", "negative"],
            "class_2": ["positive", "positive"],
            "prob_class_-1": [0.6, 0.9],
            "prob_class_2": [0.4, 0.1],
        }
    )


def test_onset_detection_preserves_signed_probability_labels() -> None:
    observations = _signed_label_observations()

    thresholded = annotate_threshold_crossings(
        observations,
        threshold_window=(-0.2, -0.2),
        threshold_quantile=0.5,
        score_column="probability_true_class",
    )

    assert thresholded["predicted_label"].tolist() == [-1, -1]
    assert thresholded["predicted_class"].tolist() == ["negative", "negative"]
    assert thresholded["onset_score"].tolist() == pytest.approx([0.6, 0.9])

    events = detect_onsets(
        observations,
        threshold_window=(-0.2, -0.2),
        threshold_quantile=0.5,
        score_column="probability_true_class",
        detection_start=0.0,
    )

    row = events.iloc[0]
    assert row["detected"]
    assert row["detection_time"] == pytest.approx(0.1)
    assert row["predicted_label_at_detection"] == -1
    assert row["predicted_class_at_detection"] == "negative"
    assert row["score_at_detection"] == pytest.approx(0.9)
    assert row["is_correct_at_detection"]
