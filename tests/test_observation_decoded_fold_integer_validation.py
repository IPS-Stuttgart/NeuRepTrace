from __future__ import annotations

import numpy as np
import pytest

from neureptrace.observations import ProbabilityObservationTable


def _decoded_fold_arguments() -> dict[str, object]:
    return {
        "probabilities": np.array([[0.2, 0.8], [0.9, 0.1]]),
        "test_labels": np.array([1, 0]),
        "predictions": np.array([1, 0]),
        "class_names": ["noise", "face"],
        "test_indices": np.array([0, 1]),
        "fold": 0,
        "decoder": "logistic",
        "backend": "sklearn",
        "emission_mode": "calibrated",
        "time": 0.15,
    }


@pytest.mark.parametrize("field", ["test_labels", "predictions", "test_indices"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        int(np.iinfo(np.int_).max) + 1,
        int(np.iinfo(np.int_).min) - 1,
    ],
)
def test_from_decoded_fold_rejects_values_outside_platform_integer_range(
    field: str,
    invalid_value: int,
) -> None:
    arguments = _decoded_fold_arguments()
    arguments[field] = np.asarray([invalid_value, 0], dtype=object)

    with pytest.raises(
        ValueError,
        match=rf"from_decoded_fold {field} values must fit the platform integer range",
    ):
        ProbabilityObservationTable.from_decoded_fold(**arguments)


@pytest.mark.parametrize("field", ["test_labels", "predictions", "test_indices"])
def test_from_decoded_fold_rejects_non_vector_integer_inputs(field: str) -> None:
    arguments = _decoded_fold_arguments()
    arguments[field] = np.asarray([[1], [0]], dtype=int)

    with pytest.raises(
        ValueError,
        match=rf"from_decoded_fold {field} must be a one-dimensional integer-valued vector",
    ):
        ProbabilityObservationTable.from_decoded_fold(**arguments)
