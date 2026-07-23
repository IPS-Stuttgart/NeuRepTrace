import pandas as pd
import pytest

from neureptrace.metrics import compare_prepost_windows, summarize_window_metric


def _metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 0.5, 1.0],
            "accuracy": [0.4, 0.6, 0.8],
        }
    )


@pytest.mark.parametrize("window", ["01", b"01", bytearray(b"01")])
def test_summarize_window_metric_rejects_string_like_windows(window: object) -> None:
    with pytest.raises(ValueError, match="window must contain exactly two values"):
        summarize_window_metric(_metric_frame(), "accuracy", window)  # type: ignore[arg-type]


@pytest.mark.parametrize("window", ["01", b"01", bytearray(b"01")])
def test_compare_prepost_windows_rejects_string_like_windows(window: object) -> None:
    with pytest.raises(ValueError, match="window must contain exactly two values"):
        compare_prepost_windows(
            _metric_frame(),
            "accuracy",
            window,  # type: ignore[arg-type]
            (0.5, 1.0),
        )
