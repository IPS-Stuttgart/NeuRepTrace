from __future__ import annotations

import pandas as pd
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols


def test_prediction_metrics_prefer_true_label_index_over_numeric_raw_label() -> None:
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subj-1",
                "true_label": 1,
                "true_label_index": 0,
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "outer_test_subject": "subj-1",
                "true_label": 2,
                "true_label_index": 1,
                "prob_class_0": 0.2,
                "prob_class_1": 0.8,
            },
        ]
    )

    metrics = all_protocols._prediction_metric_frame(predictions)

    row = metrics.iloc[0]
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    assert row["log_loss"] < 0.3
