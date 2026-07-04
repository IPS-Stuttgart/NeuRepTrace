import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics import compare_prepost_windows, summarize_window_metric


@pytest.mark.parametrize("bad_time", [True, np.bool_(False), "not-a-time", np.nan, np.inf])
def test_summarize_window_metric_rejects_invalid_time_values(bad_time):
    frame = pd.DataFrame(
        {
            "time": pd.Series([-0.1, bad_time, 0.2], dtype=object),
            "accuracy": [0.4, 0.6, 0.8],
        }
    )

    with pytest.raises(ValueError, match="time must contain only finite numeric values"):
        summarize_window_metric(frame, "accuracy", (-0.2, 0.0))


def test_compare_prepost_windows_rejects_invalid_time_values():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic"],
            "time": pd.Series([-0.2, False, 0.2], dtype=object),
            "accuracy": [0.4, 0.6, 0.8],
        }
    )

    with pytest.raises(ValueError, match="time must contain only finite numeric values"):
        compare_prepost_windows(frame, "accuracy", (-0.2, 0.0), (0.2, 0.3), group_columns=("decoder",))
