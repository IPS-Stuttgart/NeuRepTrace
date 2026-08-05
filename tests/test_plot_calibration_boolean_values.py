from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.plot_calibration import summarize_reliability_curve


def _reliability_bins() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic"],
            "time": [0.1],
            "bin": [1],
            "bin_left": [0.0],
            "bin_right": [0.5],
            "n_samples": [2],
            "sample_weight": [2.0],
            "accuracy": [0.75],
            "confidence": [0.6],
        }
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("n_samples", True, "n_samples values must be finite non-negative integers"),
        ("sample_weight", np.bool_(True), "sample_weight values must be finite and non-negative"),
        ("accuracy", False, "accuracy values must be finite probabilities"),
        ("confidence", np.bool_(False), "confidence values must be finite probabilities"),
    ],
)
def test_summarize_reliability_curve_rejects_boolean_numeric_values(column, value, message):
    bins = _reliability_bins()
    bins[column] = [value]

    with pytest.raises(ValueError, match=message):
        summarize_reliability_curve(bins)


def test_summarize_reliability_curve_preserves_numeric_zero_and_one():
    bins = _reliability_bins()
    bins["n_samples"] = [1]
    bins["sample_weight"] = [1.0]
    bins["accuracy"] = [1.0]
    bins["confidence"] = [0.0]

    curve = summarize_reliability_curve(bins)

    assert curve.loc[0, "n_samples"] == 1
    assert curve.loc[0, "accuracy"] == 1.0
    assert curve.loc[0, "confidence"] == 0.0
