import numpy as np
import pytest

from neureptrace.metrics import validate_probability_inputs


def test_validate_probability_inputs_reports_overflowing_normalized_rows_as_validation_error():
    probabilities = np.array([[1.0e308, 1.0e308]], dtype=float)

    with np.errstate(over="raise"):
        with pytest.raises(ValueError, match="probability rows must sum to one"):
            validate_probability_inputs(probabilities)


def test_validate_probability_inputs_skips_row_sum_when_normalization_is_disabled():
    probabilities = np.array([[1.0e308, 1.0e308]], dtype=float)

    with np.errstate(over="raise"):
        validated, labels = validate_probability_inputs(probabilities, require_normalized=False)

    np.testing.assert_array_equal(validated, probabilities)
    assert labels is None
