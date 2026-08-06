from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.metrics import confusion_counts, per_class_accuracy


def test_confusion_counts_rejects_group_column_colliding_with_label_output() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "prediction": [0, 1, 1, 0],
            "true_label": ["run-a", "run-a", "run-b", "run-b"],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"group_columns overlap generated confusion-count columns: \['true_label'\]",
    ):
        confusion_counts(
            frame,
            true_column="actual",
            predicted_column="prediction",
            group_columns=("true_label",),
        )


def test_per_class_accuracy_rejects_group_column_colliding_with_metric_output() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "prediction": [0, 1, 1, 0],
            "accuracy": ["fold-a", "fold-a", "fold-b", "fold-b"],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"group_columns overlap generated per-class-accuracy columns: \['accuracy'\]",
    ):
        per_class_accuracy(
            frame,
            true_column="actual",
            predicted_column="prediction",
            group_columns=("accuracy",),
        )


def test_per_class_accuracy_rejects_participant_column_colliding_with_label_output() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "prediction": [0, 1, 1, 0],
            "true_label": ["sub-01", "sub-02", "sub-01", "sub-02"],
        }
    )

    with pytest.raises(
        ValueError,
        match="participant_column conflicts with a generated per-class label column",
    ):
        per_class_accuracy(
            frame,
            true_column="actual",
            predicted_column="prediction",
            participant_column="true_label",
        )


def test_per_class_accuracy_keeps_noncolliding_custom_columns() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "prediction": [0, 1, 1, 0],
            "fold": ["fold-a", "fold-a", "fold-b", "fold-b"],
            "participant": ["sub-01", "sub-02", "sub-01", "sub-02"],
        }
    )

    summary = per_class_accuracy(
        frame,
        true_column="actual",
        predicted_column="prediction",
        participant_column="participant",
        group_columns=("fold",),
    )

    assert summary["fold"].tolist() == ["fold-a", "fold-b"]
    assert summary["true_label"].tolist() == [0, 1]
    assert summary["n_participants"].tolist() == [2, 2]
