from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.source_confidence_weighting import compute_source_confidence_weights, confidence_scores


@pytest.mark.parametrize(
    "probability_factory",
    [
        lambda: np.asarray([[0.75 + 0.1j, 0.25 - 0.1j]], dtype=np.complex128),
        lambda: np.asarray([[np.complex64(0.75 + 0.1j), 0.25 - 0.1j]], dtype=object),
        lambda: (iter(row) for row in ([0.75 + 0.1j, 0.25 - 0.1j],)),
    ],
    ids=["native-array", "object-array", "nested-one-pass"],
)
def test_source_confidence_weighting_rejects_complex_probabilities(probability_factory: Callable[[], object]) -> None:
    for function in (confidence_scores, compute_source_confidence_weights):
        with pytest.raises(ValueError, match="source_probabilities.*real-valued.*complex"):
            function(probability_factory())


def test_correct_confidence_rejects_complex_labels() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    complex_labels = np.asarray([0.0 + 1.0j, 1.0 + 0.0j])

    with pytest.raises(ValueError, match="source_labels.*real integer.*complex"):
        confidence_scores(probabilities, labels=complex_labels, mode="correct_confidence")

    with pytest.raises(ValueError, match="source_labels.*real integer.*complex"):
        compute_source_confidence_weights(
            probabilities,
            source_labels=complex_labels,
            config={"mode": "correct_confidence"},
        )
