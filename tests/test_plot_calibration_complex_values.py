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
        (
            "n_samples",
            2 + 1j,
            "Reliability-bin n_samples values must be finite non-negative integers",
        ),
        (
            "sample_weight",
            np.complex64(2 + 1j),
            "Reliability-bin sample_weight values must be finite and non-negative",
        ),
        (
            "accuracy",
            np.complex128(0.75 + 0.1j),
            "Reliability-bin accuracy values must be finite probabilities",
        ),
        (
            "confidence",
            0.6 + 0.1j,
            "Reliability-bin confidence values must be finite probabilities",
        ),
    ],
)
def test_summarize_reliability_curve_rejects_complex_values(column, value, message):
    bins = _reliability_bins()
    bins[column] = [value]

    with pytest.raises(ValueError, match=message):
        summarize_reliability_curve(bins)
