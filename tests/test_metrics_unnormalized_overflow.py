import numpy as np

from neureptrace.metrics import validate_probability_inputs


def test_validate_probability_inputs_unnormalized_scores_avoids_row_sum_overflow():
    scores = np.array([[1e308, 1e308], [1e308, 1.0]])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probabilities, labels = validate_probability_inputs(scores, require_normalized=False)

    assert labels is None
    np.testing.assert_array_equal(probabilities, scores)
