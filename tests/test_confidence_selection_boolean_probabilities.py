from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.confidence_selection import select_confident_probability_rows


@pytest.mark.parametrize(
    "probabilities",
    (
        [[True, False], [False, True]],
        np.asarray([[True, False]], dtype=bool),
        [[0.95, False]],
    ),
)
def test_boolean_probability_values_are_rejected(probabilities: object) -> None:
    with pytest.raises(ValueError, match="boolean flags"):
        select_confident_probability_rows(probabilities)  # type: ignore[arg-type]
