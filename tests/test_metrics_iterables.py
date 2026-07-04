import numpy as np
import pytest

from neureptrace.metrics import (
    brier_score_multiclass,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    top_k_accuracy,
    validate_probability_inputs,
)


def _probability_rows():
    yield (value for value in (0.8, 0.2))
    yield (value for value in (0.1, 0.9))
    yield (value for value in (0.3, 0.7))


def _labels():
    yield from (0, 1, 1)


def test_unweighted_metrics_accept_one_pass_probability_rows_and_labels():
    probabilities, labels = validate_probability_inputs(_probability_rows(), _labels())

    np.testing.assert_allclose(probabilities, np.array([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]]))
    np.testing.assert_array_equal(labels, np.array([0, 1, 1]))

    assert negative_log_likelihood(_probability_rows(), _labels()) == pytest.approx(-np.mean(np.log([0.8, 0.9, 0.7])))
    assert brier_score_multiclass(_probability_rows(), _labels()) == pytest.approx(np.mean([0.08, 0.02, 0.18]))
    assert top_k_accuracy(_probability_rows(), _labels()) == 1.0
    assert expected_calibration_error(_probability_rows(), _labels(), n_bins=2) == pytest.approx((0.2 + 0.1 + 0.3) / 3.0)
    assert reliability_bins(_probability_rows(), _labels(), n_bins=2)[1]["n_samples"] == 3


def test_unweighted_metrics_reject_boolean_inside_one_pass_probability_rows():
    def rows():
        yield (value for value in (True, False))

    with pytest.raises(ValueError, match="not boolean flags"):
        validate_probability_inputs(rows(), [0])
