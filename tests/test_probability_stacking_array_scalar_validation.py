from __future__ import annotations

import numpy as np
import pytest

from neureptrace.probability_stacking import (
    class_balanced_sample_weights,
    fit_source_oof_stacking,
    fit_stacking_weights,
)


def _probability_cube() -> np.ndarray:
    return np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.4, 0.6]],
        ],
        dtype=float,
    )


@pytest.mark.parametrize("array_value", [np.asarray(20), np.array([20])])
def test_fit_stacking_weights_rejects_array_max_iter(array_value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="max_iter must be a scalar, not an array"):
        fit_stacking_weights(_probability_cube(), [0, 1], max_iter=array_value)


def test_fit_stacking_weights_rejects_array_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate must be a scalar, not an array"):
        fit_stacking_weights(_probability_cube(), [0, 1], learning_rate=np.asarray(0.25))


def test_fit_stacking_weights_rejects_array_min_probability() -> None:
    with pytest.raises(ValueError, match="min_probability must be a scalar, not an array"):
        fit_stacking_weights(_probability_cube(), [0, 1], min_probability=np.array([1.0e-6]))


def test_fit_source_oof_stacking_rejects_array_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be a scalar, not an array"):
        fit_source_oof_stacking(
            _probability_cube(),
            [0, 1],
            candidates=("strong", "weak"),
            weighting="softmax",
            temperature=np.asarray(0.05),
        )


def test_class_balanced_sample_weights_rejects_array_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be a scalar, not an array"):
        class_balanced_sample_weights([0, 1], n_classes=np.array([2]))
