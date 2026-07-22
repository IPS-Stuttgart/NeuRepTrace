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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_labels", np.array([1.5, 0.0])),
        ("predictions", np.array([True, False])),
        ("test_indices", np.array([0.5, 1.0])),
    ],
)
def test_from_decoded_fold_rejects_lossy_integer_coercion(field: str, value: np.ndarray) -> None:
    arguments = _decoded_fold_arguments()
    arguments[field] = value

    with pytest.raises(
        ValueError,
        match=rf"{field} must be a one-dimensional sequence of finite integers, not booleans",
    ):
        ProbabilityObservationTable.from_decoded_fold(**arguments)


def test_from_decoded_fold_keeps_integer_valued_float_vectors() -> None:
    arguments = _decoded_fold_arguments()
    arguments["test_labels"] = np.array([1.0, 0.0])
    arguments["predictions"] = np.array([1.0, 0.0])
    arguments["test_indices"] = np.array([0.0, 1.0])

    table = ProbabilityObservationTable.from_decoded_fold(**arguments)

    assert table.frame["true_label"].tolist() == [1, 0]
    assert table.frame["predicted_label"].tolist() == [1, 0]
    assert table.frame["sample_index"].tolist() == [0, 1]
