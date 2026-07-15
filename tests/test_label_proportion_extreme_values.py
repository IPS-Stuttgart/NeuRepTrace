import numpy as np

from neureptrace.decoding.label_proportions import adjust_probabilities_to_label_proportions, normalize_label_proportions


def test_normalize_label_proportions_handles_large_finite_counts_without_overflow():
    with np.errstate(over="raise", invalid="raise"):
        proportions, classes = normalize_label_proportions([1.0e308, 1.0e308], classes=("rare", "standard"))

    assert classes == ("rare", "standard")
    assert np.all(np.isfinite(proportions))
    assert np.allclose(proportions, [0.5, 0.5])


def test_label_proportion_probability_rows_preserve_large_finite_ratios():
    probabilities = np.asarray(
        [
            [1.0e308, 1.0],
            [1.0, 1.0e308],
        ]
    )

    with np.errstate(over="raise", invalid="raise"):
        result = adjust_probabilities_to_label_proportions(
            probabilities,
            [1.0e308, 1.0e308],
            classes=("rare", "standard"),
            tol=1.0e-12,
        )

    assert result.converged
    assert np.all(np.isfinite(result.probabilities))
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.probabilities[0, 0] > 1.0 - 1.0e-9
    assert result.probabilities[0, 1] < 1.0e-9
    assert result.probabilities[1, 1] > 1.0 - 1.0e-9
    assert result.probabilities[1, 0] < 1.0e-9
