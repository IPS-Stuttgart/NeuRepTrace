import numpy as np
import pandas as pd
import pytest

from neureptrace.semantic_stages import detect_stable_stages


def _time_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic"],
            "emission_mode": ["calibrated"],
            "true_class": ["animate"],
            "time": [0.1],
            "posterior_true_class_mean": [0.8],
            "viterbi_match_rate": [0.9],
            "n_sequences": [3],
        }
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time", True),
        ("time", np.bool_(False)),
        ("time", 0.1 + 0.2j),
        ("posterior_true_class_mean", np.complex128(0.8 + 0.1j)),
        ("viterbi_match_rate", np.array(0.9 + 0.1j)),
    ],
)
def test_detect_stable_stages_rejects_boolean_and_complex_numeric_values(column: str, value: object):
    time_summary = _time_summary()
    time_summary[column] = pd.Series([value], dtype=object)

    with pytest.raises(ValueError, match=rf"{column} values must be numeric"):
        detect_stable_stages(time_summary)


def test_detect_stable_stages_preserves_valid_real_numeric_values():
    time_summary = _time_summary()

    stages = detect_stable_stages(
        time_summary,
        posterior_threshold=0.7,
        match_threshold=0.7,
        min_duration=0.0,
    )

    assert stages.loc[0, "start_time"] == pytest.approx(0.1)
    assert stages.loc[0, "peak_posterior_true_class"] == pytest.approx(0.8)
