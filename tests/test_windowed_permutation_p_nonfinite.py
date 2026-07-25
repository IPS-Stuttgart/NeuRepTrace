import numpy as np
import pytest

from neureptrace.decoding.windowed import permutation_p_from_accuracy


def test_permutation_p_ignores_nonfinite_null_scores():
    permutation_scores = np.array([0.1, np.nan, np.inf, -np.inf, 0.8])

    assert permutation_p_from_accuracy(0.75, permutation_scores) == pytest.approx(2.0 / 3.0)


def test_permutation_p_is_undefined_without_finite_null_scores():
    assert np.isnan(permutation_p_from_accuracy(0.75, np.array([np.nan, np.inf, -np.inf])))
