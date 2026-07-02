from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_temperature import apply_temperature, negative_log_likelihood


def test_apply_temperature_accepts_one_pass_probability_rows() -> None:
    rows = (row for row in ([0.75, 0.25], [0.20, 0.80]))

    scaled = apply_temperature(rows, temperature=1.0)

    assert scaled.shape == (2, 2)
    assert np.allclose(scaled.sum(axis=1), 1.0)


def test_negative_log_likelihood_accepts_one_pass_label_indices() -> None:
    labels = (item for item in [1, 0])

    value = negative_log_likelihood([[0.25, 0.75], [0.8, 0.2]], labels)

    assert np.isclose(value, -np.mean(np.log([0.75, 0.8])))
