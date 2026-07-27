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
    "probabilities",
    [
        np.array(
            [
                [0.8 + 0.1j, 0.2 - 0.1j],
                [0.2 + 0.3j, 0.8 - 0.3j],
            ],
            dtype=np.complex128,
        ),
        np.array(
            [
                [np.asarray(0.8 + 0.1j), np.asarray(0.2 - 0.1j)],
                [0.2, 0.8],
            ],
            dtype=object,
        ),
        [
            [np.complex128(0.8 + 0.1j), 0.2 - 0.1j],
            [0.2, 0.8],
        ],
    ],
)
def test_from_decoded_fold_rejects_complex_probabilities(probabilities: object) -> None:
    with pytest.raises(ValueError, match="probabilities .* complex"):
        ProbabilityObservationTable.from_decoded_fold(
            probabilities=probabilities,
            **_BASE_DECODED_FOLD_ARGS,
        )


@pytest.mark.parametrize("field_name", ["test_labels", "predictions", "test_indices"])
@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0 + 1.0j, 1.0 + 0.0j], dtype=np.complex128),
        np.array([np.asarray(0.0 + 1.0j), np.asarray(1.0 + 0.0j)], dtype=object),
        [np.complex128(0.0 + 1.0j), 1.0 + 0.0j],
    ],
)
def test_from_decoded_fold_rejects_complex_integer_vectors(field_name: str, values: object) -> None:
    kwargs = dict(_BASE_DECODED_FOLD_ARGS)
    kwargs[field_name] = values

    with pytest.raises(ValueError, match=fr"from_decoded_fold {field_name} .* complex"):
        ProbabilityObservationTable.from_decoded_fold(
            probabilities=np.array([[0.8, 0.2], [0.2, 0.8]]),
            **kwargs,
        )
