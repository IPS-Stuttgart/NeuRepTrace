from __future__ import annotations

import pandas as pd

from neureptrace.onset_detection import annotate_threshold_crossings


def test_onset_infers_predictions_from_numeric_probability_suffixes():
    frame = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "sequence_id": [0, 0],
            "time": [-0.1, 0.1],
            "class_1": ["left", "left"],
            "class_2": ["right", "right"],
            "prob_class_1": [0.15, 0.80],
            "prob_class_2": [0.85, 0.20],
        }
    )

    thresholded = annotate_threshold_crossings(
        frame,
        threshold_window=(-0.1, -0.1),
        threshold_quantile=0.5,
    )

    assert thresholded["predicted_label"].tolist() == [2, 1]
    assert thresholded["predicted_class"].tolist() == ["right", "left"]


def test_probability_true_class_scores_use_probability_suffix_labels():
    frame = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "sequence_id": [0, 0],
            "time": [-0.1, 0.1],
            "true_label": [2, 1],
            "prob_class_1": [0.15, 0.70],
            "prob_class_2": [0.85, 0.30],
        }
    )

    thresholded = annotate_threshold_crossings(
        frame,
        threshold_window=(-0.1, 0.1),
        threshold_quantile=0.5,
        score_column="probability_true_class",
    )

    assert thresholded["onset_score"].tolist() == [0.85, 0.70]
