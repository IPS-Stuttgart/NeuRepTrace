from decimal import Decimal

import numpy as np

from neureptrace import _sign_flip_scalar_controls_patch as scalar_controls
from neureptrace import inference, paired_stats


def _reference_sign_flip_t_statistics(effects: np.ndarray, *, n_permutations: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_subjects = effects.shape[0]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n_subjects))
    means = signs @ effects / n_subjects
    sum_squares = np.sum(effects**2, axis=0)
    variances = (sum_squares[None, :] - n_subjects * means**2) / (n_subjects - 1)
    sem = np.sqrt(np.maximum(variances, 0.0) / n_subjects)
    return np.divide(means, sem, out=np.zeros_like(means), where=sem > 0)


def test_sign_flip_integer_validators_preserve_values_above_float64_exact_range():
    value = 2**53 + 1

    assert scalar_controls._validate_random_state(value) == value
    assert scalar_controls._validate_positive_permutation_count(value) == value
    assert inference._validate_positive_permutation_count(value) == value
    assert paired_stats._validate_random_state(value) == value


def test_sign_flip_integer_validators_preserve_exact_decimal_representations():
    value = 2**53 + 1
    representations = (
        str(value),
        f"{value}.0",
        "9.007199254740993e15",
        Decimal(value),
        Decimal(f"{value}.0"),
    )

    for representation in representations:
        assert scalar_controls._validate_random_state(representation) == value
        assert scalar_controls._validate_positive_permutation_count(representation) == value
        assert inference._validate_positive_permutation_count(representation) == value
        assert paired_stats._validate_random_state(representation) == value


def test_sign_flip_statistics_use_exact_large_integer_seed():
    effects = np.asarray(
        [
            [0.2, 1.1, -0.4],
            [0.7, -0.3, 0.8],
            [-1.2, 0.5, 1.7],
            [0.4, -1.5, 0.2],
        ],
        dtype=float,
    )
    seed = 2**53 + 1
    n_permutations = 17

    actual = inference._sign_flip_t_statistics(
        effects,
        n_permutations=n_permutations,
        random_state=seed,
    )
    exact_expected = _reference_sign_flip_t_statistics(
        effects,
        n_permutations=n_permutations,
        random_state=seed,
    )
    rounded_expected = _reference_sign_flip_t_statistics(
        effects,
        n_permutations=n_permutations,
        random_state=2**53,
    )

    np.testing.assert_allclose(actual, exact_expected)
    assert not np.array_equal(actual, rounded_expected)


def test_sign_flip_statistics_use_exact_large_decimal_string_seed():
    effects = np.asarray(
        [
            [0.2, 1.1, -0.4],
            [0.7, -0.3, 0.8],
            [-1.2, 0.5, 1.7],
            [0.4, -1.5, 0.2],
        ],
        dtype=float,
    )
    seed = 2**53 + 1
    n_permutations = 17

    actual = inference._sign_flip_t_statistics(
        effects,
        n_permutations=n_permutations,
        random_state=str(seed),
    )
    exact_expected = _reference_sign_flip_t_statistics(
        effects,
        n_permutations=n_permutations,
        random_state=seed,
    )
    rounded_expected = _reference_sign_flip_t_statistics(
        effects,
        n_permutations=n_permutations,
        random_state=2**53,
    )

    np.testing.assert_allclose(actual, exact_expected)
    assert not np.array_equal(actual, rounded_expected)
