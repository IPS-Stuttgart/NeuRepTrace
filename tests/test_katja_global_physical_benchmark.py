from __future__ import annotations

import inspect

import numpy as np
import pytest

from neureptrace.katja_global_physical_benchmark import (
    _validate_trial_label_uniqueness,
    restrict_probabilities_to_calibration_classes,
)


def test_restrict_probabilities_uses_only_calibration_classes() -> None:
    probabilities = np.asarray(
        [
            [
                [0.05, 0.70, 0.10, 0.10, 0.05],
                [0.60, 0.25, 0.05, 0.05, 0.05],
                [0.05, 0.70, 0.10, 0.10, 0.05],
                [0.05, 0.70, 0.10, 0.10, 0.05],
            ]
        ],
        dtype=float,
    )
    model_classes = np.asarray([1, 2, 3, 4, 5])
    calibration_labels = np.asarray([[1, 3, 4, 5], [3, 4, 5, 1]])

    allowed, restricted = restrict_probabilities_to_calibration_classes(
        probabilities,
        model_classes,
        calibration_labels,
    )

    np.testing.assert_array_equal(allowed, np.asarray([1, 3, 4, 5]))
    np.testing.assert_allclose(restricted.sum(axis=2), np.ones((1, 4)))
    # The excluded physical class 2 has the largest original probability in
    # three rows, but cannot be predicted after the calibration-only mask.
    predictions = allowed[np.argmax(restricted, axis=2)]
    assert 2 not in predictions


def test_restrict_probabilities_rejects_missing_target_class() -> None:
    with pytest.raises(ValueError, match="not represented exactly once"):
        restrict_probabilities_to_calibration_classes(
            np.full((1, 4, 4), 0.25),
            np.asarray([1, 2, 3, 4]),
            np.asarray([1, 2, 3, 5]),
        )


def test_target_class_mask_api_does_not_accept_evaluation_labels() -> None:
    parameters = inspect.signature(
        restrict_probabilities_to_calibration_classes
    ).parameters
    assert "evaluation_labels" not in parameters
    assert "target_labels" not in parameters


def test_trial_label_uniqueness_audits_physical_fingers() -> None:
    _validate_trial_label_uniqueness(
        np.asarray([[1, 2, 3, 4], [2, 3, 4, 5]]),
        expected_classes=4,
        name="physical labels",
    )
    with pytest.raises(ValueError, match="unique physical fingers"):
        _validate_trial_label_uniqueness(
            np.asarray([[1, 1, 3, 4]]),
            expected_classes=4,
            name="physical labels",
        )
