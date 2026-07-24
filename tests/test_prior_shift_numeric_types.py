from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.prior_shift import adapt_probabilities_for_prior_shift


@pytest.mark.parametrize(
    "probabilities",
    [
        [[True, False], [False, True]],
        np.asarray([[True, False], [False, True]], dtype=bool),
        np.asarray([[0.5 + 0.1j, 0.5 - 0.1j]], dtype=complex),
    ],
)
def test_prior_shift_rejects_non_real_probability_values(probabilities: object) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        adapt_probabilities_for_prior_shift(probabilities)


@pytest.mark.parametrize(
    ("prior_name", "prior"),
    [
        ("source_prior", [True, False]),
        ("initial_target_prior", np.asarray([0.5 + 0.1j, 0.5 - 0.1j])),
        ("target_prior", np.asarray([0.5 + 0.1j, 0.5 - 0.1j])),
    ],
)
def test_prior_shift_rejects_non_real_prior_values(prior_name: str, prior: object) -> None:
    with pytest.raises(ValueError, match=prior_name):
        adapt_probabilities_for_prior_shift([[0.6, 0.4], [0.4, 0.6]], **{prior_name: prior})


@pytest.mark.parametrize(
    ("control", "value"),
    [
        ("max_iter", np.complex128(2 + 1j)),
        ("tol", np.complex128(1e-8 + 1j)),
        ("smoothing", np.complex128(0 + 1j)),
        ("damping", np.complex128(1 + 1j)),
        ("epsilon", np.complex128(1e-12 + 1j)),
    ],
)
def test_prior_shift_rejects_numpy_complex_scalar_controls(control: str, value: object) -> None:
    with pytest.raises(ValueError, match=control):
        adapt_probabilities_for_prior_shift([[0.6, 0.4], [0.4, 0.6]], **{control: value})


def test_prior_shift_still_accepts_real_probability_and_prior_inputs() -> None:
    result = adapt_probabilities_for_prior_shift(
        [[0.6, 0.4], [0.4, 0.6]],
        source_prior=np.asarray([0.5, 0.5]),
        target_prior=np.asarray([0.75, 0.25]),
    )

    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.target_prior, [0.75, 0.25])
