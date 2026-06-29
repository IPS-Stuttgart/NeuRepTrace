from __future__ import annotations

import numpy as np
import pytest

from neureptrace.observations import ProbabilityObservationTable


def _base_call_args() -> dict[str, object]:
    return {
        "probabilities": np.array([[0.2, 0.8]]),
        "test_labels": np.array([1]),
        "predictions": np.array([1]),
        "class_names": ["noise", "face"],
        "test_indices": np.array([0]),
        "fold": 0,
        "decoder": "logistic",
        "backend": "sklearn",
        "emission_mode": "calibrated",
        "time": 0.15,
    }


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("test_labels", np.array([-1])),
        ("test_labels", np.array([2])),
        ("predictions", np.array([-1])),
        ("predictions", np.array([2])),
    ],
)
def test_from_decoded_fold_rejects_invalid_class_indices(field: str, values: np.ndarray) -> None:
    args = _base_call_args()
    args[field] = values

    with pytest.raises(ValueError, match=field):
        ProbabilityObservationTable.from_decoded_fold(**args)


def test_from_decoded_fold_rejects_negative_test_indices() -> None:
    args = _base_call_args()
    args["test_indices"] = np.array([-1])

    with pytest.raises(ValueError, match="test_indices must contain non-negative row indices"):
        ProbabilityObservationTable.from_decoded_fold(**args)


def test_from_decoded_fold_rejects_original_index_lookup_overrun() -> None:
    args = _base_call_args()
    args["test_indices"] = np.array([1])
    args["original_indices"] = np.array([10])

    with pytest.raises(ValueError, match="outside original_indices"):
        ProbabilityObservationTable.from_decoded_fold(**args)
