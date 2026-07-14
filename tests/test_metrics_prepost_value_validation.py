from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics import compare_prepost_windows, summarize_window_metric


@pytest.mark.parametrize("bad_value", ["not-a-number", True, np.inf, -np.inf])
def test_summarize_window_metric_rejects_invalid_metric_values(bad_value: object) -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "accuracy": pd.Series([0.5, bad_value], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="accuracy must contain only finite numeric values or missing values"):
        summarize_window_metric(frame, "accuracy", (0.0, 0.1))


def test_summarize_window_metric_preserves_missing_value_semantics() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "accuracy": [0.5, np.nan],
        }
    )

    summary = summarize_window_metric(frame, "accuracy", (0.0, 0.1))

    assert summary.loc[0, "n_rows"] == 1
    assert summary.loc[0, "accuracy_mean"] == pytest.approx(0.5)


def test_compare_prepost_windows_rejects_invalid_post_metric_values() -> None:
    frame = pd.DataFrame(
        {
            "time": [-0.1, 0.1],
            "accuracy": pd.Series([0.5, "invalid"], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="accuracy must contain only finite numeric values or missing values"):
        compare_prepost_windows(frame, "accuracy", (-0.1, -0.1), (0.1, 0.1))
