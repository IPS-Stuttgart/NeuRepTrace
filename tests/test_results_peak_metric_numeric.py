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


def test_peak_metric_rows_preserves_columns_matching_internal_sort_names() -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "participant": ["s1", "s1"],
            "time": [0.1, 0.2],
            "accuracy": [0.8, 0.9],
            "_peak_metric_numeric": ["loser metric metadata", "winner metric metadata"],
            "_peak_time_numeric": ["loser time metadata", "winner time metadata"],
            "_peak_distance_to_prefer_time": ["loser distance metadata", "winner distance metadata"],
        }
    )

    peaks = peak_metric_rows(frame, "accuracy", ("decoder", "participant"))

    assert peaks.loc[0, "_peak_metric_numeric"] == "winner metric metadata"
    assert peaks.loc[0, "_peak_time_numeric"] == "winner time metadata"
    assert peaks.loc[0, "_peak_distance_to_prefer_time"] == "winner distance metadata"
    assert peaks.loc[0, "peak_distance_to_prefer_time"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("accuracy", "bad"),
        ("accuracy", np.nan),
        ("accuracy", True),
        ("accuracy", 0.9 + 0.1j),
        ("time", "bad"),
        ("time", np.inf),
        ("time", True),
        ("time", np.complex128(0.1 + 0.2j)),
    ],
)
def test_peak_metric_rows_rejects_invalid_selection_values(column: str, value: object) -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "participant": ["s1", "s1"],
            "time": ["0.1", "0.2"],
            "accuracy": ["0.9", "0.8"],
        },
        dtype=object,
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
