from __future__ import annotations

import numpy as np
import pytest

from neureptrace.observations import ProbabilityObservationTable


_BASE_DECODED_FOLD_ARGS = {
    "test_labels": np.array([0, 1]),
    "predictions": np.array([0, 1]),
    "class_names": ["zero", "one"],
    "test_indices": np.array([0, 1]),
    "fold": 0,
    "decoder": "logistic",
    "backend": "sklearn",
    "emission_mode": "calibrated",
    "time": 0.15,
}


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([[0.8, 0.4], [0.2, 0.8]]), "sum to 1.0"),
        (np.array([[np.nan, 1.0], [0.2, 0.8]]), "finite"),
        (np.array([[-0.1, 1.1], [0.2, 0.8]]), "non-negative"),
        (np.array([[True, False], [False, True]]), "not boolean"),
    ],
)
def test_from_decoded_fold_rejects_invalid_probability_values(probabilities: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProbabilityObservationTable.from_decoded_fold(
            probabilities=probabilities,
            **_BASE_DECODED_FOLD_ARGS,
        )
