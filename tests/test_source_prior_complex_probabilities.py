from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior


@pytest.mark.parametrize(
    "probabilities",
    [
        np.asarray([[0.8 + 0.1j, 0.2 - 0.1j]], dtype=np.complex128),
        np.asarray([[0.8 + 0.1j, 0.2]], dtype=object),
    ],
    ids=["complex-array", "object-array"],
)
def test_source_prior_rejects_complex_probability_arrays(probabilities: np.ndarray) -> None:
    with pytest.raises(ValueError, match="probabilities.*complex"):
        adjust_probabilities_to_source_prior(probabilities, source_labels=[0, 1], classes=[0, 1])


def test_source_prior_rejects_one_pass_complex_probability_rows() -> None:
    probabilities: Iterable[list[complex]] = (row for row in [[0.8 + 0.1j, 0.2 - 0.1j]])

    with pytest.raises(ValueError, match="probabilities.*complex"):
        adjust_probabilities_to_source_prior(probabilities, source_labels=[0, 1], classes=[0, 1])
