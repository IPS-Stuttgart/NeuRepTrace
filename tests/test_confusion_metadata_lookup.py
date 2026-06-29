from __future__ import annotations

import numpy as np
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


def test_confusion_metadata_lookup_accepts_array_valued_labels() -> None:
    cat_label = np.asarray(["cat", 1], dtype=object)
    dog_label = np.asarray(["dog", 1], dtype=object)
    predictions = pd.DataFrame(
        {
            "true_label": [cat_label, dog_label],
            "predicted_label": [dog_label, cat_label],
        }
    )
    metadata = pd.DataFrame(
        {
            "label": [cat_label, dog_label],
            "semantic_category": ["animal", "animal"],
        }
    )

    pair_summary = confusion_pair_summary(predictions, metadata_frame=metadata)
    category_matrix = confusion_category_matrix(predictions, metadata_frame=metadata, category_columns=("semantic_category",))

    assert len(pair_summary) == 1
    assert pair_summary.loc[0, "label_a"] == ("cat", 1)
    assert pair_summary.loc[0, "label_b"] == ("dog", 1)
    assert pair_summary.loc[0, "label_a_semantic_category"] == "animal"
    assert pair_summary.loc[0, "label_b_semantic_category"] == "animal"
    assert bool(pair_summary.loc[0, "same_semantic_category"])
    assert int(pair_summary.loc[0, "total_confusions"]) == 2

    assert len(category_matrix) == 1
    assert int(category_matrix.loc[0, "count"]) == 2
    assert bool(category_matrix.loc[0, "same_category"])
