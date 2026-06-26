from __future__ import annotations

import pandas as pd

from neureptrace.metrics import confusion_category_matrix, confusion_pair_summary


def test_confusion_metadata_lookup_does_not_truncate_fractional_labels() -> None:
    predictions = pd.DataFrame({"true_label": [1.5], "predicted_label": [2.5]})
    metadata = pd.DataFrame(
        {
            "label": [1, 2],
            "semantic_category": ["integer-one", "integer-two"],
        }
    )

    pair_summary = confusion_pair_summary(predictions, metadata_frame=metadata)

    assert len(pair_summary) == 1
    assert "label_a_semantic_category" not in pair_summary.columns
    assert "label_b_semantic_category" not in pair_summary.columns
    assert confusion_category_matrix(predictions, metadata_frame=metadata, category_columns=("semantic_category",)).empty
