from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.metrics import confusion_pair_summary, per_class_accuracy


def test_per_class_accuracy_treats_matching_missing_labels_as_correct() -> None:
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p2", "p3", "p4"],
            "true_label": [np.nan, pd.NA, pd.NaT, "cat"],
            "predicted_label": [np.nan, pd.NA, pd.NaT, "dog"],
        }
    )

    summary = per_class_accuracy(predictions, participant_column="participant")

    missing = summary.loc[pd.isna(summary["true_label"])].iloc[0]
    assert missing["n_trials"] == 3
    assert missing["n_correct"] == 3
    assert missing["accuracy"] == 1.0
    assert missing["n_participants"] == 3


def test_confusion_pair_summary_ignores_matching_missing_labels() -> None:
    predictions = pd.DataFrame(
        {
            "true_label": [np.nan, pd.NA, pd.NaT, "cat"],
            "predicted_label": [np.nan, pd.NA, pd.NaT, "cat"],
        }
    )

    summary = confusion_pair_summary(predictions)

    assert summary.empty
    assert {"label_a", "label_b", "total_confusions"}.issubset(summary.columns)
