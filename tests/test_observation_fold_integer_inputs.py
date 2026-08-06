from __future__ import annotations

import numpy as np
import pytest

from neureptrace.observations import ProbabilityObservationTable


def _decoded_fold_kwargs() -> dict[str, object]:
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
    ("argument", "invalid_values", "message"),
    [
        ("test_labels", np.array([0.5, 1.0]), "finite integer values"),
        ("test_labels", np.array([False, True]), "boolean flags"),
        ("predictions", np.array([0.0 + 1.0j, 1.0 + 0.0j]), "complex values"),
        ("test_indices", np.array([0.5, 1.0]), "finite integer values"),
        ("test_indices", np.array([False, True]), "boolean flags"),
    ],
)
def test_from_decoded_fold_rejects_values_that_integer_casting_would_change(
    argument: str,
    invalid_values: np.ndarray,
    message: str,
) -> None:
    kwargs = _decoded_fold_kwargs()
    kwargs[argument] = invalid_values

    with pytest.raises(ValueError, match=message):
        ProbabilityObservationTable.from_decoded_fold(**kwargs)


def test_from_decoded_fold_accepts_integer_valued_float_vectors() -> None:
    kwargs = _decoded_fold_kwargs()
    kwargs["test_labels"] = np.array([1.0, 0.0])
    kwargs["predictions"] = np.array([1.0, 0.0])
    kwargs["test_indices"] = np.array([0.0, 1.0])

    table = ProbabilityObservationTable.from_decoded_fold(**kwargs)

    assert table.frame["true_label"].tolist() == [1, 0]
    assert table.frame["predicted_label"].tolist() == [1, 0]
    assert table.frame["sample_index"].tolist() == [0, 1]
