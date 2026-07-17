import numpy as np
import pandas as pd

from neureptrace.results import summarize_metric_table


def test_summarize_metric_table_handles_constant_extreme_finite_values() -> None:
    maximum = np.finfo(float).max
    frame = pd.DataFrame(
        {
            "decoder": ["stacked", "stacked"],
            "score": [maximum, maximum],
            "chance": [maximum, maximum],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        row = summarize_metric_table(
            frame,
            "score",
            "decoder",
            chance_column="chance",
        ).iloc[0]

    assert row["score_mean"] == maximum
    assert row["score_median"] == maximum
    assert row["score_std"] == 0.0
    assert row["score_sem"] == 0.0
    assert row["chance_mean"] == maximum
    assert row["score_minus_chance_mean"] == 0.0


def test_summarize_metric_table_rescales_extreme_dispersion_separately() -> None:
    maximum = np.finfo(float).max
    frame = pd.DataFrame(
        {
            "decoder": ["stacked", "stacked"],
            "score": [maximum, -maximum],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        row = summarize_metric_table(frame, "score", "decoder").iloc[0]

    assert row["score_mean"] == 0.0
    assert row["score_median"] == 0.0
    assert row["score_std"] == maximum
    assert row["score_sem"] == maximum
