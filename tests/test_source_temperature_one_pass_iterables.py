from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_temperature import apply_temperature, fit_source_temperature_scaling


def _row_generators(rows):
    return ((value for value in row) for row in rows)


def test_apply_temperature_accepts_one_pass_probability_rows() -> None:
    scaled = apply_temperature(_row_generators([[0.7, 0.3], [0.2, 0.8]]), temperature=1.0)

    assert np.allclose(scaled, [[0.7, 0.3], [0.2, 0.8]])


def test_fit_source_temperature_scaling_accepts_one_pass_probability_rows() -> None:
    result = fit_source_temperature_scaling(
        source_probabilities=_row_generators([[0.70, 0.30], [0.65, 0.35], [0.35, 0.65], [0.30, 0.70]]),
        source_labels=(label for label in [0, 0, 1, 1]),
        test_probabilities=_row_generators([[0.60, 0.40], [0.25, 0.75]]),
        classes=[0, 1],
        config={"temperatures": [0.5, 1.0, 2.0]},
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert set(result.source_losses) == {0.5, 1.0, 2.0}
