from __future__ import annotations

import pandas as pd

from neureptrace.metrics import confusion_pair_summary


def test_confusion_pair_summary_keeps_schema_without_errors() -> None:
    predictions = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "true_label": [0, 1],
            "predicted_label": [0, 1],
        }
    )

    summary = confusion_pair_summary(predictions, group_columns=("decoder",))

    assert summary.empty
    assert {
        "decoder",
        "label_a",
        "label_b",
        "a_to_b_count",
        "b_to_a_count",
        "total_confusions",
        "pair_confusion_lift",
    } <= set(summary.columns)


def test_confusion_pair_summary_empty_schema_includes_metadata_columns() -> None:
    predictions = pd.DataFrame(
        {
            "true_stimulus": [1, 2],
            "predicted_stimulus": [1, 2],
        }
    )
    metadata = pd.DataFrame(
        {
            "stimulus": [1, 2],
            "semantic_category": ["animal", "object"],
        }
    )

    summary = confusion_pair_summary(
        predictions,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        metadata_frame=metadata,
        metadata_label_columns=("stimulus",),
        label_prefix="stimulus",
    )

    assert summary.empty
    assert {
        "stimulus_a",
        "stimulus_b",
        "stimulus_a_semantic_category",
        "stimulus_b_semantic_category",
        "same_semantic_category",
    } <= set(summary.columns)
