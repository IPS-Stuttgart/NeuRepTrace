from __future__ import annotations

import pandas as pd

from neureptrace.decoding.temporal_generalization import summarize_temporal_generalization_matrix


def test_summary_compares_each_accuracy_with_its_own_chance_level() -> None:
    # Different class availability gives each participant a distinct chance baseline.
    rows = pd.DataFrame(
        {
            "participant": ["S01", "S02"],
            "decoder": ["toy", "toy"],
            "accuracy": [0.6, 0.4],
            "chance_accuracy": [0.5, 0.25],
        }
    )

    summary = summarize_temporal_generalization_matrix(rows, group_columns="decoder")

    assert summary.loc[0, "chance_accuracy"] == 0.375
    assert summary.loc[0, "chance_percent"] == 37.5
    assert summary.loc[0, "above_chance_count"] == 2
