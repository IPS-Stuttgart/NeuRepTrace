import numpy as np
import pytest

from neureptrace.decoding.label_proportions import (
    adjust_probabilities_to_label_proportions,
    adjust_probability_blocks_to_label_proportions,
    normalize_label_proportions,
)


@pytest.mark.parametrize(
    "target_proportions",
    [
        [True, False],
        np.asarray([np.bool_(True), np.bool_(False)]),
        {"target": True, "standard": 3},
    ],
)
def test_normalize_label_proportions_rejects_boolean_values(target_proportions):
    with pytest.raises(ValueError, match="not boolean flags"):
        normalize_label_proportions(target_proportions)


def test_adjust_probabilities_to_label_proportions_rejects_boolean_target_prior():
    with pytest.raises(ValueError, match="not boolean flags"):
        adjust_probabilities_to_label_proportions([[0.6, 0.4], [0.4, 0.6]], [True, False])


def test_blockwise_label_proportion_calibration_rejects_boolean_target_prior():
    with pytest.raises(ValueError, match="not boolean flags"):
        adjust_probability_blocks_to_label_proportions(
            [[0.6, 0.4], [0.4, 0.6]],
            ["run1", "run1"],
            {"run1": [np.bool_(True), 0]},
        )
