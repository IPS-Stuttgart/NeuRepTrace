from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols


def test_prediction_metric_frame_uses_group_local_probability_columns() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "true_label": "cat",
                "predicted_label": "cat",
                "prob_class_cat": 0.9,
                "prob_class_dog": 0.1,
                "prob_class_bird": np.nan,
                "prob_class_fish": np.nan,
            },
            {
                "outer_test_subject": "subj-1",
                "true_label": "dog",
                "predicted_label": "dog",
                "prob_class_cat": 0.2,
                "prob_class_dog": 0.8,
                "prob_class_bird": np.nan,
                "prob_class_fish": np.nan,
            },
            {
                "outer_test_subject": "subj-2",
                "true_label": "fish",
                "predicted_label": "fish",
                "prob_class_cat": np.nan,
                "prob_class_dog": np.nan,
                "prob_class_bird": 0.3,
                "prob_class_fish": 0.7,
            },
            {
                "outer_test_subject": "subj-2",
                "true_label": "bird",
                "predicted_label": "bird",
                "prob_class_cat": np.nan,
                "prob_class_dog": np.nan,
                "prob_class_bird": 0.6,
                "prob_class_fish": 0.4,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions).set_index("outer_test_subject")

    assert metrics.loc["subj-1", "accuracy"] == pytest.approx(1.0)
    assert metrics.loc["subj-1", "balanced_accuracy"] == pytest.approx(1.0)
    assert metrics.loc["subj-2", "accuracy"] == pytest.approx(1.0)
    assert metrics.loc["subj-2", "balanced_accuracy"] == pytest.approx(1.0)
    assert np.isfinite(metrics.loc["subj-1", "log_loss"])
    assert np.isfinite(metrics.loc["subj-2", "log_loss"])


def test_prediction_metric_frame_falls_back_for_probabilityless_groups() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "with-probs",
                "true_label": 0,
                "predicted_label": 0,
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "outer_test_subject": "labels-only",
                "true_label": "target",
                "predicted_label": "target",
                "prob_class_0": np.nan,
                "prob_class_1": np.nan,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions).set_index("outer_test_subject")

    assert metrics.loc["with-probs", "accuracy"] == pytest.approx(1.0)
    assert metrics.loc["labels-only", "accuracy"] == pytest.approx(1.0)
    assert np.isnan(metrics.loc["labels-only", "top2_accuracy"])
