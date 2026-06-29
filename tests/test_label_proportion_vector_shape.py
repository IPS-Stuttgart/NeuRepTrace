import numpy as np
import pytest

from neureptrace.decoding.label_proportions import (
    adjust_probabilities_to_label_proportions,
    adjust_probability_blocks_to_label_proportions,
    normalize_label_proportions,
)


_PROBABILITIES = [[0.6, 0.4], [0.4, 0.6]]
_CLASSES = ("target", "standard")


@pytest.mark.parametrize(
    "target_proportions",
    [
        [[0.25, 0.75]],
        np.asarray([[0.25, 0.75]]),
        np.asarray([[1.0], [3.0]]),
    ],
)
def test_normalize_label_proportions_rejects_multidimensional_target_prior(target_proportions):
    with pytest.raises(ValueError, match="one-dimensional sequence"):
        normalize_label_proportions(target_proportions, classes=_CLASSES)


def test_adjust_probabilities_to_label_proportions_rejects_multidimensional_target_prior():
    with pytest.raises(ValueError, match="one-dimensional sequence"):
        adjust_probabilities_to_label_proportions(
            _PROBABILITIES,
            np.asarray([[0.5, 0.5]]),
            classes=_CLASSES,
        )


def test_blockwise_label_proportion_calibration_rejects_multidimensional_block_prior():
    with pytest.raises(ValueError, match="one-dimensional sequence"):
        adjust_probability_blocks_to_label_proportions(
            _PROBABILITIES,
            ["run1", "run1"],
            {"run1": np.asarray([[0.5, 0.5]])},
            classes=_CLASSES,
        )
