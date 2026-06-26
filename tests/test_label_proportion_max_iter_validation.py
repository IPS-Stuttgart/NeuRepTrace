import numpy as np
import pytest

from neureptrace.decoding.label_proportions import adjust_probabilities_to_label_proportions


@pytest.mark.parametrize("max_iter", [0, 1.5, float("inf"), float("nan"), None, True, np.bool_(True)])
def test_label_proportion_calibration_rejects_invalid_max_iter(max_iter):
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        adjust_probabilities_to_label_proportions(
            [[0.6, 0.4], [0.4, 0.6]],
            [0.5, 0.5],
            max_iter=max_iter,
        )
