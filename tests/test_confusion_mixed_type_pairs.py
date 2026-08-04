from __future__ import annotations

import pandas as pd

from neureptrace.metrics import confusion_pair_summary


def test_confusion_pair_summary_coalesces_mixed_type_sort_key_ties() -> None:
    frame = pd.DataFrame(
        {
            "true_label": pd.Series([1, "1"], dtype=object),
            "predicted_label": pd.Series(["1", 1], dtype=object),
        }
    )

    summary = confusion_pair_summary(frame)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["label_a"] == 1
    assert row["label_b"] == "1"
    assert row["a_to_b_count"] == 1
    assert row["b_to_a_count"] == 1
    assert row["total_confusions"] == 2
    assert row["symmetric_confusion_count"] == 1
