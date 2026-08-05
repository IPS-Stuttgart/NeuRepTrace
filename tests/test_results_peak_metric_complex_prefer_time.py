from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import peak_metric_rows


def _tied_metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "participant": ["s1", "s1"],
            "time": [-0.2, 0.2],
            "accuracy": [0.9, 0.9],
        }
    )


@pytest.mark.parametrize(
    "prefer_time",
    [
        np.complex64(0.1 + 1.0j),
        np.complex128(-0.1 + 2.0j),
    ],
)
def test_peak_metric_rows_rejects_complex_numpy_preferred_time(prefer_time: object) -> None:
    with pytest.raises(ValueError, match="prefer_time must be a finite numeric value"):
        peak_metric_rows(
            _tied_metric_frame(),
            "accuracy",
            ("decoder", "participant"),
            prefer_time=prefer_time,
        )


def test_peak_metric_rows_accepts_real_numpy_preferred_time() -> None:
    peaks = peak_metric_rows(
        _tied_metric_frame(),
        "accuracy",
        ("decoder", "participant"),
        prefer_time=np.float64(0.1),
    )

    assert peaks.loc[0, "time"] == pytest.approx(0.2)
    assert peaks.loc[0, "peak_distance_to_prefer_time"] == pytest.approx(0.1)
