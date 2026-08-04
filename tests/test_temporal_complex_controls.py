import numpy as np
import pandas as pd
import pytest

from neureptrace.temporal_model import build_state_trace, fit_sticky_switching_model


_COMPLEX_VALUES = [
    np.complex64(3.0 + 1.0j),
    np.complex128(3.0 + 1.0j),
    np.asarray(np.complex64(3.0 + 1.0j)),
]


@pytest.mark.parametrize("value", _COMPLEX_VALUES)
def test_fit_sticky_switching_model_rejects_complex_grid_sizes(value):
    sequence = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=float)

    with pytest.raises(ValueError, match="stay_grid_size must be an integer"):
        fit_sticky_switching_model([sequence], stay_grid_size=value)


@pytest.mark.parametrize("value", _COMPLEX_VALUES)
def test_build_state_trace_rejects_complex_stay_probabilities(value):
    frame = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "sequence_id": [0, 0],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "time": [0.0, 0.1],
            "prob_class_0": [0.8, 0.7],
            "prob_class_1": [0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="stay_probability"):
        build_state_trace(
            frame,
            stay_probability=value,
            class_names=["left", "right"],
            prob_columns=["prob_class_0", "prob_class_1"],
        )
