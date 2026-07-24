from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics import compare_prepost_windows, summarize_window_metric


def _metric_frame() -> pd.DataFrame:
    return pd.DataFrame({"time": [-0.1, 0.1], "accuracy": [0.5, 0.6]})


@pytest.mark.parametrize("endpoint", [np.complex64(-0.1 + 2j), np.complex128(0.1 + 3j)])
def test_window_metrics_reject_numpy_complex_endpoints(endpoint: object) -> None:
    with pytest.raises(ValueError, match="window start must be a finite numeric value"):
        summarize_window_metric(_metric_frame(), "accuracy", (endpoint, 0.1))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="window stop must be a finite numeric value"):
        compare_prepost_windows(_metric_frame(), "accuracy", (-0.1, -0.1), (0.1, endpoint))  # type: ignore[arg-type]


def test_summarize_window_metric_rejects_complex_time_values() -> None:
    frame = pd.DataFrame(
        {
            "time": np.asarray([-0.1 + 1j, 0.1 + 2j]),
            "accuracy": [0.5, 0.6],
        }
    )

    with pytest.raises(ValueError, match="time must contain only finite numeric values"):
        summarize_window_metric(frame, "accuracy", (-0.1, 0.1))


def test_summarize_window_metric_rejects_complex_metric_values() -> None:
    frame = pd.DataFrame(
        {
            "time": [-0.1, 0.1],
            "accuracy": np.asarray([0.5 + 1j, 0.6 + 2j]),
        }
    )

    with pytest.raises(ValueError, match="accuracy must contain only finite numeric values or missing values"):
        summarize_window_metric(frame, "accuracy", (-0.1, 0.1))
