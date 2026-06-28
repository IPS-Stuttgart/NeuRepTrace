from __future__ import annotations

import pandas as pd
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols


def test_prediction_metric_frame_accepts_named_probability_columns() -> None:
    predictions = pd.DataFrame(
        [
            {"outer_test_subject": "s01", "true_label": "alpha", "prob_class_alpha": 0.90, "prob_class_beta": 0.10},
            {"outer_test_subject": "s01", "true_label": "beta", "prob_class_alpha": 0.20, "prob_class_beta": 0.80},
            {"outer_test_subject": "s01", "true_label": "alpha", "prob_class_alpha": 0.70, "prob_class_beta": 0.30},
            {"outer_test_subject": "s01", "true_label": "beta", "prob_class_alpha": 0.05, "prob_class_beta": 0.95},
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    assert metrics.shape[0] == 1
    row = metrics.iloc[0]
    assert row["outer_test_subject"] == "s01"
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    assert row["top2_accuracy"] == pytest.approx(1.0)


def test_prediction_metric_frame_maps_sparse_numeric_probability_columns() -> None:
    predictions = pd.DataFrame(
        [
            {"outer_test_subject": "s01", "true_label_index": 10, "prob_class_10": 0.90, "prob_class_2": 0.10},
            {"outer_test_subject": "s01", "true_label_index": 2, "prob_class_10": 0.20, "prob_class_2": 0.80},
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    assert metrics.iloc[0]["accuracy"] == pytest.approx(1.0)
    assert metrics.iloc[0]["balanced_accuracy"] == pytest.approx(1.0)


def test_prediction_metric_frame_maps_index_labels_to_named_probability_columns() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "s01",
                "true_label_index": 0,
                "class_0": "alpha",
                "class_1": "beta",
                "prob_class_alpha": 0.85,
                "prob_class_beta": 0.15,
            },
            {
                "outer_test_subject": "s01",
                "true_label_index": 1,
                "class_0": "alpha",
                "class_1": "beta",
                "prob_class_alpha": 0.25,
                "prob_class_beta": 0.75,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    assert metrics.iloc[0]["accuracy"] == pytest.approx(1.0)
    assert metrics.iloc[0]["balanced_accuracy"] == pytest.approx(1.0)
