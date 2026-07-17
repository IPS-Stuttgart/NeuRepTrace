from __future__ import annotations

import numpy as np

from neureptrace import inference


def _extreme_effects() -> np.ndarray:
    limit = np.finfo(float).max
    return np.array(
        [
            [limit, limit, limit, limit],
            [limit, -limit, limit, limit],
            [limit, limit, -limit, limit],
            [limit, -limit, -limit, -limit],
        ],
        dtype=float,
    )


def test_t_statistic_is_scale_invariant_at_float_limit() -> None:
    effects = _extreme_effects()
    normalized = effects / np.finfo(float).max

    with np.errstate(over="raise", invalid="raise"):
        actual = inference._t_statistic(effects)
        expected = inference._t_statistic(normalized)

    np.testing.assert_array_equal(expected, np.array([np.inf, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(actual, expected)


def test_permutation_t_statistics_are_stable_at_float_limit() -> None:
    effects = _extreme_effects()
    normalized = effects / np.finfo(float).max

    with np.errstate(over="raise", invalid="raise"):
        actual = inference._sign_flip_t_statistics(
            effects,
            n_permutations=32,
            random_state=7,
        )
        expected = inference._sign_flip_t_statistics(
            normalized,
            n_permutations=32,
            random_state=7,
        )

    assert not np.isnan(actual).any()
    np.testing.assert_array_equal(actual, expected)
