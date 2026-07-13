from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_temperature import apply_temperature, fit_source_temperature_scaling


def test_source_temperature_normalizes_extreme_finite_rows_without_overflow() -> None:
    scores = np.asarray(
        [
            [1.0e308, 1.0e308],
            [1.0e308, 1.0e307],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        transformed = apply_temperature(scores, temperature=1.0)
        fitted = fit_source_temperature_scaling(
            source_probabilities=scores,
            source_labels=[0, 1],
            test_probabilities=scores,
            classes=[0, 1],
            config={"temperatures": [1.0]},
        )

    expected = np.asarray([[0.5, 0.5], [10.0 / 11.0, 1.0 / 11.0]])
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(fitted.probabilities, expected, rtol=1.0e-6)
    assert fitted.temperature == 1.0
    assert np.isfinite(fitted.source_losses[1.0])


def test_source_temperature_preserves_probability_mass_floor() -> None:
    with pytest.raises(ValueError, match="positive probability mass"):
        apply_temperature([[1.0e-13, 1.0e-13]], temperature=1.0, epsilon=1.0e-12)

    accepted = apply_temperature([[6.0e-13, 6.0e-13]], temperature=1.0, epsilon=1.0e-12)
    np.testing.assert_allclose(accepted, [[0.5, 0.5]])
