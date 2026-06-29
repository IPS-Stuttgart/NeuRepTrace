import pytest

from neureptrace.decoding.label_proportions import (
    adjust_probabilities_to_label_proportions,
    adjust_probability_blocks_to_label_proportions,
)


_PROBABILITIES = [[0.6, 0.4], [0.4, 0.6]]
_TARGET_PRIOR = [0.5, 0.5]


@pytest.mark.parametrize("tol", [None, [0.0], {"tol": 0.0}])
def test_label_proportion_calibration_rejects_malformed_tol(tol):
    with pytest.raises(ValueError, match="tol must be non-negative and finite"):
        adjust_probabilities_to_label_proportions(
            _PROBABILITIES,
            _TARGET_PRIOR,
            tol=tol,
        )


@pytest.mark.parametrize("epsilon", [None, [1e-12], {"epsilon": 1e-12}])
def test_label_proportion_calibration_rejects_malformed_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        adjust_probabilities_to_label_proportions(
            _PROBABILITIES,
            _TARGET_PRIOR,
            epsilon=epsilon,
        )


def test_blockwise_label_proportion_calibration_rejects_malformed_epsilon():
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        adjust_probability_blocks_to_label_proportions(
            _PROBABILITIES,
            ["run1", "run1"],
            {"run1": _TARGET_PRIOR},
            epsilon=None,
        )
