from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import validate_sample_weight
from neureptrace.metrics.weighted import weighted_top_k_accuracy


@pytest.mark.parametrize(
    "sample_weight",
    [
        np.asarray([np.complex128(1.0 + 2.0j), 1.0], dtype=object),
        np.asarray([[np.complex128(1.0 + 2.0j)], [1.0]], dtype=object),
    ],
)
def test_validate_sample_weight_rejects_numpy_complex_object_cells(sample_weight: np.ndarray) -> None:
    with pytest.raises(ValueError, match="real-valued weights"):
        validate_sample_weight(sample_weight, 2)


def test_weighted_metric_rejects_one_pass_complex_weights() -> None:
    probabilities = np.asarray([[0.7, 0.3], [0.4, 0.6]])
    labels = np.asarray([0, 1])
    sample_weight = (value for value in [np.complex128(1.0 + 2.0j), 1.0])

    with pytest.raises(ValueError, match="real-valued weights"):
        weighted_top_k_accuracy(probabilities, labels, sample_weight)
