import numpy as np
import pytest

from neureptrace.decoding import parse_c_grid, predict_emission_probabilities


class MessyProbabilityModel:
    classes_ = np.array(["a", "b", "c"])

    def predict_proba(self, features):
        return np.array(
            [
                [2.0, 2.0, 0.0],
                [np.nan, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [-1.0, 2.0, 1.0],
                [np.inf, 3.0, np.inf],
            ]
        )

    def predict(self, features):
        return np.array(["a", "b", "c", "a", "b"])


def test_predict_emission_probabilities_sanitizes_messy_probability_rows():
    probabilities = predict_emission_probabilities(MessyProbabilityModel(), np.zeros((5, 2)))

    assert probabilities.shape == (5, 3)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[0], [0.5, 0.5, 0.0])
    assert np.allclose(probabilities[1], [0.0, 0.5, 0.5])
    assert np.allclose(probabilities[2], [0.0, 0.0, 1.0])
    assert np.allclose(probabilities[3], [0.0, 2.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(probabilities[4], [0.5, 0.0, 0.5])


def test_parse_c_grid_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="positive finite"):
        parse_c_grid("0.1,nan,1.0")

    with pytest.raises(ValueError, match="positive finite"):
        parse_c_grid([0.1, np.inf])
