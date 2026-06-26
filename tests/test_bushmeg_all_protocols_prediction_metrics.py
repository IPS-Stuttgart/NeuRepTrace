from __future__ import annotations

import pandas as pd
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols


def test_prediction_metric_frame_uses_true_label_index_for_string_labels() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "trial_index": 0,
                "true_label": "face",
                "true_label_index": 0,
                "predicted_label": "face",
                "predicted_label_index": 0,
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "outer_test_subject": "subj-1",
                "trial_index": 1,
                "true_label": "scrambled",
                "true_label_index": 1,
                "predicted_label": "scrambled",
                "predicted_label_index": 1,
                "prob_class_0": 0.2,
                "prob_class_1": 0.8,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    row = metrics.iloc[0]
    assert row["outer_test_subject"] == "subj-1"
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    assert row["top2_accuracy"] == pytest.approx(1.0)
    assert row["top3_accuracy"] == pytest.approx(1.0)
    assert row["log_loss"] < 0.3


def test_prediction_metric_frame_can_map_class_columns_when_label_index_missing() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "true_label": "left",
                "class_0": "left",
                "class_1": "right",
                "prob_class_0": 0.7,
                "prob_class_1": 0.3,
            },
            {
                "outer_test_subject": "subj-1",
                "true_label": "right",
                "class_0": "left",
                "class_1": "right",
                "prob_class_0": 0.1,
                "prob_class_1": 0.9,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    assert metrics.loc[0, "accuracy"] == pytest.approx(1.0)
    assert metrics.loc[0, "balanced_accuracy"] == pytest.approx(1.0)
