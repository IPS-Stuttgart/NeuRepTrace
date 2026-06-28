from __future__ import annotations

import pandas as pd
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols


def test_label_only_metric_recompute_prefers_explicit_class_positions() -> None:
    true_col = "true_" + "label"
    true_index_col = true_col + "_index"
    predicted_index_col = "predicted_" + "label_index"
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "subject-a",
                "fold_index": 1,
                true_col: 10,
                true_index_col: 0,
                predicted_index_col: 0,
            },
            {
                "outer_test_subject": "subject-a",
                "fold_index": 1,
                true_col: 20,
                true_index_col: 1,
                predicted_index_col: 1,
            },
        ]
    )

    row = all_protocols._prediction_metric_frame(predictions).iloc[0]

    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    assert pd.isna(row["top2_accuracy"])
    assert pd.isna(row["log_loss"])
