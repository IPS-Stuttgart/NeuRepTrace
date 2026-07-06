from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_grid import score_probability_shape


def test_probability_shape_accepts_one_pass_probability_rows():
    probability_rows = ((value for value in row) for row in ((0.70, 0.30), (0.25, 0.75)))

    score, terms = score_probability_shape(probability_rows, active_classes=2)

    assert np.isfinite(score)
    assert terms["active_fraction"] == 1.0
    assert terms["confidence"] == pytest.approx(0.725)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[True, False], [False, True]]),
        [[True, False], [False, True]],
        ((value for value in row) for row in ((True, False), (False, True))),
    ],
)
def test_probability_shape_rejects_boolean_probability_values(probabilities):
    with pytest.raises(ValueError, match="boolean"):
        score_probability_shape(probabilities)


def test_probability_shape_rejects_negative_probability_values():
    with pytest.raises(ValueError, match="non-negative"):
        score_probability_shape([[0.8, 0.2], [-0.1, 1.1]])
