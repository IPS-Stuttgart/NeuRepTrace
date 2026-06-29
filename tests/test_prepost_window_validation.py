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


@pytest.mark.parametrize("endpoint", [np.asarray(0.0), np.array([0.0]), np.asarray(True), np.array([True])])
def test_summarize_window_metric_rejects_array_valued_start_endpoint(endpoint: object) -> None:
    with pytest.raises(ValueError, match="window start must be a finite numeric scalar"):
        summarize_window_metric(_metric_frame(), "accuracy", (endpoint, 0.1))  # type: ignore[arg-type]


@pytest.mark.parametrize("endpoint", [np.asarray(0.1), np.array([0.1]), np.asarray(True), np.array([True])])
def test_summarize_window_metric_rejects_array_valued_stop_endpoint(endpoint: object) -> None:
    with pytest.raises(ValueError, match="window stop must be a finite numeric scalar"):
        summarize_window_metric(_metric_frame(), "accuracy", (0.0, endpoint))  # type: ignore[arg-type]


@pytest.mark.parametrize("endpoint", [np.asarray(0.0), np.array([0.0]), np.asarray(True), np.array([True])])
def test_compare_prepost_windows_rejects_array_valued_window_endpoints(endpoint: object) -> None:
    with pytest.raises(ValueError, match="window start must be a finite numeric scalar"):
        compare_prepost_windows(_metric_frame(), "accuracy", (endpoint, 0.0), (0.1, 0.1))  # type: ignore[arg-type]


def test_window_metrics_still_accept_numpy_numeric_scalars() -> None:
    summary = summarize_window_metric(_metric_frame(), "accuracy", (np.float64(0.0), np.float64(0.1)))

    assert summary["n_rows"].iloc[0] == 2
    assert summary["accuracy_mean"].iloc[0] == pytest.approx(0.55)


def test_compare_prepost_windows_rejects_non_sized_window() -> None:
    with pytest.raises(ValueError, match="window must contain exactly two values"):
        compare_prepost_windows(_metric_frame(), "accuracy", 0.0, (0.0, 0.1))  # type: ignore[arg-type]
