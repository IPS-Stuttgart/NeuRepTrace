import pandas as pd
import pytest

from neureptrace.results import summarize_metric_table


def test_summarize_metric_table_preserves_tuple_valued_single_group():
    first_group = ("session-1", "run-1")
    second_group = ("session-2", "run-1")
    frame = pd.DataFrame(
        {
            "condition": [first_group, first_group, second_group],
            "accuracy": [0.6, 0.8, 0.7],
        }
    )

    summary = summarize_metric_table(frame, "accuracy", "condition")

    assert summary["condition"].tolist() == [first_group, second_group]
    assert summary["n_rows"].tolist() == [2, 1]
    assert summary["accuracy_mean"].tolist() == pytest.approx([0.7, 0.7])
