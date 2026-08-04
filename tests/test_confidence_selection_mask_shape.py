from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neureptrace.decoding.confidence import accepted_probability_rows, select_confident_rows


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.bool_(True),
        np.asarray([[True, False]], dtype=bool),
        np.asarray([[True], [False]], dtype=bool),
        np.asarray([[True, False], [False, True]], dtype=bool),
    ],
)
def test_accepted_probability_rows_rejects_non_vector_selection_masks(bad_mask: object) -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    selection = select_confident_rows(probabilities)
    malformed_selection = replace(selection, accepted_mask=bad_mask)

    with pytest.raises(ValueError, match="selection mask must be one-dimensional"):
        accepted_probability_rows(probabilities, selection=malformed_selection)


def test_accepted_probability_rows_accepts_one_pass_boolean_mask() -> None:
    probabilities = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=float)
    selection = select_confident_rows(probabilities)
    one_pass_selection = replace(selection, accepted_mask=(value for value in [True, False]))

    accepted = accepted_probability_rows(probabilities, selection=one_pass_selection)

    np.testing.assert_allclose(accepted, [[1.0, 0.0]])
