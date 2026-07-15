from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.metrics import (
    confusion_category_enrichment,
    confusion_category_matrix,
    confusion_pair_summary,
)


@pytest.mark.parametrize("group_value", ["logistic", ("logistic", 1)])
def test_confusion_summaries_preserve_single_group_values(group_value):
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "decoder": [group_value] * 6,
            "true_label": [1, 1, 2, 1, 3, 4],
            "predicted_label": [2, 2, 1, 1, 4, 3],
        }
    )
    metadata = pd.DataFrame(
        {
            "label": [1, 2, 3, 4],
            "semantic_category": ["animal", "animal", "object", "object"],
        }
    )

    summaries = [
        confusion_pair_summary(
            predictions,
            group_columns=("decoder",),
            participant_column="participant",
            metadata_frame=metadata,
        ),
        confusion_category_enrichment(
            predictions,
            metadata_frame=metadata,
            category_columns=("semantic_category",),
            group_columns=("decoder",),
            participant_column="participant",
            n_permutations=0,
        ),
        confusion_category_matrix(
            predictions,
            metadata_frame=metadata,
            category_columns=("semantic_category",),
            group_columns=("decoder",),
            participant_column="participant",
        ),
    ]

    for summary in summaries:
        assert not summary.empty
        assert all(value == group_value for value in summary["decoder"].tolist())
