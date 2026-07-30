from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.plot_calibration import summarize_reliability_curve


def _reliability_bins() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [0.1, 0.2],
            "bin": [1, 1],
            "bin_left": [0.0, 0.0],
            "bin_right": [0.5, 0.5],
            "n_samples": [1, 1],
            "accuracy": [1.0, 0.0],
            "confidence": [0.9, 0.1],
        }
    )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("accuracy", np.nan),
        ("accuracy", 1.1),
        ("confidence", np.inf),
        ("confidence", "invalid"),
    ],
)
def test_reliability_curve_rejects_invalid_positive_mass_metrics(column, invalid_value):
    bins = _reliability_bins()
    if isinstance(invalid_value, str):
        bins[column] = bins[column].astype(object)
    bins.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=rf"{column} values must be finite probabilities in \[0, 1\]"):
        summarize_reliability_curve(bins)


def test_reliability_curve_allows_missing_metrics_for_zero_mass_rows():
    bins = _reliability_bins()
    bins.loc[0, "n_samples"] = 0
    bins.loc[0, ["accuracy", "confidence"]] = np.nan

    curve = summarize_reliability_curve(bins)

    assert curve.loc[0, "n_samples"] == 1
    assert curve.loc[0, "accuracy"] == 0.0
    assert curve.loc[0, "confidence"] == 0.1
