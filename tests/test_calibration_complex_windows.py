import numpy as np
import pandas as pd
import pytest

from neureptrace.calibration import summarize_calibration_metrics


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [-0.05, 0.15],
            "accuracy_mean": [0.50, 0.60],
            "log_loss_mean": [0.70, 0.66],
            "brier_mean": [0.50, 0.47],
            "ece_mean": [0.09, 0.06],
            "n_subjects": [5, 5],
        }
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        np.complex64(0.1 + 2.0j),
        np.complex128(0.1 + 2.0j),
        np.asarray(0.1 + 2.0j),
        np.asarray(0.1 + 2.0j, dtype=object),
    ],
)
def test_summarize_calibration_metrics_rejects_complex_window_endpoints(endpoint):
    with pytest.raises(ValueError, match="finite real numeric values, not complex"):
        summarize_calibration_metrics(_summary_frame(), effect_window=(endpoint, 0.2))


def test_summarize_calibration_metrics_preserves_one_shot_window_iterables():
    effect_window = (endpoint for endpoint in (0.1, 0.2))

    summary = summarize_calibration_metrics(_summary_frame(), effect_window=effect_window)

    assert summary["decoder"].tolist() == ["logistic"]
    assert summary["effect_ece_mean"].tolist() == [0.06]
