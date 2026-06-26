import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics import compare_prepost_windows, summarize_window_metric


def _metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "accuracy": [0.5, 0.6],
        }
    )


@pytest.mark.parametrize(
    "window",
    [
        (np.nan, 0.1),
        (0.0, np.inf),
        (False, 0.1),
        ("bad", 0.1),
    ],
)
def test_summarize_window_metric_rejects_malformed_window_endpoints(window: object) -> None:
    with pytest.raises(ValueError, match="window .* finite numeric value"):
        summarize_window_metric(_metric_frame(), "accuracy", window)  # type: ignore[arg-type]


def test_compare_prepost_windows_rejects_non_sized_window() -> None:
    with pytest.raises(ValueError, match="window must contain exactly two values"):
        compare_prepost_windows(_metric_frame(), "accuracy", 0.0, (0.0, 0.1))  # type: ignore[arg-type]
