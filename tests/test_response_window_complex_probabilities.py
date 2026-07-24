from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.response_window_ensemble import _response_window_rows
from neureptrace.temporal_model import _validate_probability_matrix


def test_probability_matrix_rejects_complex_values_before_float_coercion() -> None:
    probabilities = np.asarray([[0.8 + 0.1j, 0.2 - 0.1j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="real-valued probabilities"):
        _validate_probability_matrix(probabilities)


def test_response_window_rejects_complex_probabilities_before_dataframe_cast() -> None:
    observations = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "sample_index": [0, 0],
            "time": [0.1, 0.2],
            "true_label": [0, 0],
            "class_0": ["zero", "zero"],
            "class_1": ["one", "one"],
            "prob_class_0": [0.8 + 0.1j, 0.7 + 0.2j],
            "prob_class_1": [0.2 - 0.1j, 0.3 - 0.2j],
        }
    )

    with pytest.raises(ValueError, match="real-valued values"):
        _response_window_rows(
            observations,
            requested_times=(0.1, 0.2),
            mode="uniform",
            combine="probability_mean",
            weight_grid_step=0.5,
            output_time=0.15,
        )
