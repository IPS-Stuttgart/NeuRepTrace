from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.metrics import confusion_counts, confusion_pair_summary


def test_confusion_counts_rejects_reused_true_and_predicted_column() -> None:
    frame = pd.DataFrame({"label": [0, 1]})

    with pytest.raises(ValueError, match="column roles must reference distinct columns"):
        confusion_counts(frame, true_column="label", predicted_column="label")


def test_confusion_counts_rejects_duplicate_physical_required_columns() -> None:
    frame = pd.DataFrame(
        [[0, 1, 1]],
        columns=["true_label", "true_label", "predicted_label"],
    )

    with pytest.raises(ValueError, match="ambiguous duplicate required columns"):
        confusion_counts(frame)


def test_confusion_pair_summary_rejects_group_role_collision() -> None:
    frame = pd.DataFrame(
        {
            "true_label": [0, 1],
            "predicted_label": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="column roles must reference distinct columns"):
        confusion_pair_summary(frame, group_columns=["true_label"])
