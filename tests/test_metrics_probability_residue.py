import numpy as np

from neureptrace.metrics import brier_score_multiclass, validate_probability_inputs


def test_validate_probability_inputs_sanitizes_tolerated_negative_residue():
    probabilities = np.array([[1.0 + 5.0e-8, -5.0e-8], [2.0e-8, 1.0 - 2.0e-8]])

    cleaned, labels = validate_probability_inputs(probabilities, np.array([0, 1]))

    assert labels.tolist() == [0, 1]
    assert np.all(cleaned >= 0.0)
    np.testing.assert_allclose(cleaned.sum(axis=1), 1.0)


def test_brier_score_uses_sanitized_probability_residue():
    probabilities = np.array([[1.0 + 5.0e-8, -5.0e-8]])

    assert brier_score_multiclass(probabilities, np.array([0])) == 0.0
