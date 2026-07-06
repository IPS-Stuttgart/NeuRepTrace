from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.label_proportions import adjust_probabilities_to_label_proportions, normalize_label_proportions


_PROBABILITIES = np.asarray([[0.6, 0.4], [0.4, 0.6]], dtype=float)


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"max_iter": np.asarray(True)}, "max_iter"),
        ({"max_iter": np.asarray([2])}, "max_iter"),
        ({"tol": np.asarray(True)}, "tol"),
        ({"tol": np.asarray([1.0e-9])}, "tol"),
        ({"epsilon": np.asarray(True)}, "epsilon"),
        ({"epsilon": np.asarray([1.0e-12])}, "epsilon"),
    ],
)
def test_label_proportion_calibration_rejects_boolean_or_vector_numeric_controls(kwargs: dict[str, object], pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        adjust_probabilities_to_label_proportions(
            _PROBABILITIES,
            [1.0, 1.0],
            classes=("rare", "standard"),
            **kwargs,
        )


def test_label_proportion_calibration_accepts_scalar_numpy_numeric_controls() -> None:
    result = adjust_probabilities_to_label_proportions(
        _PROBABILITIES,
        [1.0, 1.0],
        classes=("rare", "standard"),
        max_iter=np.asarray(25),
        tol=np.asarray(1.0e-8),
        epsilon=np.asarray(1.0e-12),
    )

    assert result.classes == ("rare", "standard")
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "target_proportions",
    [
        {"rare": np.asarray(True), "standard": 1.0},
        [np.asarray(True), 1.0],
        np.asarray([True, False]),
    ],
)
def test_normalize_label_proportions_rejects_boolean_numpy_array_values(target_proportions: object) -> None:
    with pytest.raises(ValueError, match="target_proportions"):
        normalize_label_proportions(target_proportions, classes=("rare", "standard"))


def test_normalize_label_proportions_accepts_scalar_numpy_numeric_values() -> None:
    proportions, classes = normalize_label_proportions(
        {"rare": np.asarray(1.0), "standard": np.asarray(3.0)},
        classes=("rare", "standard"),
    )

    assert classes == ("rare", "standard")
    assert np.allclose(proportions, [0.25, 0.75])
