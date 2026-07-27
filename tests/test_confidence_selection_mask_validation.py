from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neureptrace.decoding.confidence import accepted_probability_rows, select_confident_rows


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.asarray([0, 1], dtype=int),
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray(["", "selected"], dtype=object),
        [False, 2],
    ],
)
def test_accepted_probability_rows_rejects_non_boolean_selection_masks(bad_mask: object) -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    selection = select_confident_rows(probabilities)
    malformed_selection = replace(selection, accepted_mask=bad_mask)

    with pytest.raises(ValueError, match="selection mask must contain only boolean values"):
        accepted_probability_rows(probabilities, selection=malformed_selection)


def test_accepted_probability_rows_accepts_object_boolean_selection_mask() -> None:
    probabilities = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=float)
    selection = select_confident_rows(probabilities)
    object_boolean_selection = replace(
        selection,
        accepted_mask=np.asarray([np.bool_(True), False], dtype=object),
    )

    accepted = accepted_probability_rows(probabilities, selection=object_boolean_selection)

    np.testing.assert_allclose(accepted, [[1.0, 0.0]])
