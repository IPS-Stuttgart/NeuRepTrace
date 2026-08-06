from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.metrics.prepost import compare_prepost_windows, summarize_window_metric


def test_summarize_window_metric_stabilizes_repeated_extreme_values() -> None:
    maximum = np.finfo(float).max
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.0, 0.1],
            "condition": ["positive", "positive", "negative", "negative"],
            "metric": [maximum, maximum, -maximum, -maximum],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        result = summarize_window_metric(
            frame,
            "metric",
            (0.0, 0.1),
            group_columns=("condition",),
        ).set_index("condition")

    assert result.loc["positive", "metric_mean"] == maximum
    assert result.loc["negative", "metric_mean"] == -maximum
    assert result.loc["positive", "metric_std"] == 0.0
    assert result.loc["negative", "metric_std"] == 0.0
    assert result.loc["positive", "metric_sem"] == 0.0
    assert result.loc["negative", "metric_sem"] == 0.0


def test_compare_prepost_windows_stabilizes_equal_extreme_windows() -> None:
    maximum = np.finfo(float).max
    frame = pd.DataFrame(
        {
            "time": [-1.0, -0.5, 0.5, 1.0],
            "metric": [maximum, maximum, maximum, maximum],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        result = compare_prepost_windows(
            frame,
            "metric",
            (-1.0, -0.5),
            (0.5, 1.0),
        )

    assert result.loc[0, "metric_pre_mean"] == maximum
    assert result.loc[0, "metric_post_mean"] == maximum
    assert result.loc[0, "metric_pre_std"] == 0.0
    assert result.loc[0, "metric_post_std"] == 0.0
    assert result.loc[0, "metric_post_minus_pre"] == 0.0
