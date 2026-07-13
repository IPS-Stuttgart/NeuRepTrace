from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.metrics import compare_prepost_windows, summarize_window_metric


def _frame(group_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            group_column: ["group-a", "group-a", "group-b", "group-b"],
            "time": [-0.2, 0.2, -0.2, 0.2],
            "accuracy": [0.4, 0.8, 0.5, 0.9],
        }
    )


@pytest.mark.parametrize("group_column", ["window_start", "n_rows", "accuracy_mean"])
def test_summary_rejects_group_columns_that_collide_with_generated_fields(group_column):
    with pytest.raises(ValueError, match="overlap generated summary columns"):
        summarize_window_metric(
            _frame(group_column),
            "accuracy",
            (-0.2, 0.2),
            group_columns=(group_column,),
        )


@pytest.mark.parametrize(
    "group_column",
    ["pre_window_start", "n_post_rows", "accuracy_post_minus_pre"],
)
def test_comparison_rejects_group_columns_that_collide_after_renaming(group_column):
    with pytest.raises(ValueError, match="overlap generated comparison columns"):
        compare_prepost_windows(
            _frame(group_column),
            "accuracy",
            (-0.2, -0.2),
            (0.2, 0.2),
            group_columns=(group_column,),
        )
