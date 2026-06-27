from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import peak_metric_rows


def test_peak_metric_rows_sorts_csv_loaded_metric_values_numerically() -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic"],
            "participant": ["s1", "s1", "s1"],
            "time": ["0.1", "0.2", "0.3"],
            "accuracy": ["9", "10", "8"],
        }
    )

    peaks = peak_metric_rows(frame, "accuracy", ("decoder", "participant"), prefer_time="0.0")

    assert peaks.loc[0, "time"] == "0.2"
    assert peaks.loc[0, "accuracy"] == "10"
    assert peaks.loc[0, "peak_distance_to_prefer_time"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("accuracy", "bad"),
        ("accuracy", np.nan),
        ("accuracy", True),
        ("time", "bad"),
        ("time", np.inf),
        ("time", True),
    ],
)
def test_peak_metric_rows_rejects_invalid_selection_values(column: str, value: object) -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "participant": ["s1", "s1"],
            "time": ["0.1", "0.2"],
            "accuracy": ["0.9", "0.8"],
        }
    )
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=f"{column} must contain only finite numeric values"):
        peak_metric_rows(frame, "accuracy", ("decoder", "participant"))


def test_peak_metric_rows_rejects_nonfinite_preferred_time() -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic"],
            "participant": ["s1"],
            "time": ["0.1"],
            "accuracy": ["0.9"],
        }
    )

    with pytest.raises(ValueError, match="prefer_time must be a finite numeric value"):
        peak_metric_rows(frame, "accuracy", ("decoder", "participant"), prefer_time=np.inf)
