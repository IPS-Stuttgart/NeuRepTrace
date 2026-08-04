from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.prior_shift import (
    adapt_probabilities_for_prior_shift,
    reweight_probabilities_by_prior,
)


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
@pytest.mark.parametrize("name", ["source_prior", "initial_target_prior", "target_prior"])
def test_adaptation_rejects_matrix_shaped_priors(name: str, shape: tuple[int, int]) -> None:
    kwargs = {name: np.asarray([0.6, 0.4], dtype=float).reshape(shape)}

    with pytest.raises(ValueError, match=rf"{name} must be one-dimensional"):
        adapt_probabilities_for_prior_shift(
            [[0.7, 0.3], [0.2, 0.8]],
            **kwargs,
        )


@pytest.mark.parametrize("name", ["source_prior", "target_prior"])
def test_reweighting_rejects_matrix_shaped_priors(name: str) -> None:
    kwargs = {
        "source_prior": [0.5, 0.5],
        "target_prior": [0.6, 0.4],
    }
    kwargs[name] = np.asarray([[0.5, 0.5]], dtype=float)

    with pytest.raises(ValueError, match=rf"{name} must be one-dimensional"):
        reweight_probabilities_by_prior([[0.5, 0.5]], **kwargs)


def test_vector_priors_remain_valid() -> None:
    result = adapt_probabilities_for_prior_shift(
        [[0.7, 0.3], [0.2, 0.8]],
        source_prior=np.asarray([0.5, 0.5]),
        target_prior=(value for value in [0.6, 0.4]),
    )

    np.testing.assert_allclose(result.target_prior, [0.6, 0.4])
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
