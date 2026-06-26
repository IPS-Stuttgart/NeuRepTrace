from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.label_shift import (
    LABEL_SHIFT_CATEGORY,
    adapt_label_shift_probabilities,
    adjust_probabilities_to_prior,
    estimate_target_prior_bbse,
    estimate_target_prior_em,
    normalize_label_shift_method,
    soft_confusion_matrix,
)


def test_em_label_shift_estimates_unlabeled_target_prior() -> None:
    target_probabilities = np.asarray([[0.98, 0.02]] * 8 + [[0.02, 0.98]] * 2)

    result = adapt_label_shift_probabilities(
        target_probabilities,
        method="em",
        source_prior=[0.5, 0.5],
        classes=["major", "minor"],
    )

    assert result.method == "em"
    assert result.classes == ("major", "minor")
    assert result.target_prior[0] > 0.75
    assert result.target_prior[1] < 0.25
    assert result.metadata["label_shift_protocol_category"] == LABEL_SHIFT_CATEGORY
    assert result.metadata["label_shift_uses_target_probabilities"] is True
    assert result.metadata["label_shift_uses_target_labels"] is False
    assert result.metadata["label_shift_valid_for_unlabeled_target_adaptation"] is True
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_bbse_label_shift_uses_source_validation_confusion() -> None:
    source_validation_probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    source_validation_labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    target_probabilities = np.asarray([[0.85, 0.15]] * 2 + [[0.15, 0.85]] * 8)

    result = adapt_label_shift_probabilities(
        target_probabilities,
        method="bbse",
        source_validation_probabilities=source_validation_probabilities,
        source_validation_labels=source_validation_labels,
        classes=["a", "b"],
    )

    assert result.method == "bbse"
    assert result.confusion_matrix is not None
    assert result.n_iterations == 0
    assert result.converged is True
    assert result.target_prior[1] > result.target_prior[0]
    assert result.metadata["label_shift_uses_source_validation_probabilities"] is True
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_bbse_em_initializes_em_from_confusion_prior() -> None:
    source_validation_probabilities = np.asarray([[0.95, 0.05], [0.9, 0.1], [0.1, 0.9], [0.05, 0.95]])
    source_validation_labels = np.asarray([0, 0, 1, 1])
    target_probabilities = np.asarray([[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 7)

    result = adapt_label_shift_probabilities(
        target_probabilities,
        method="bbse-em",
        source_validation_probabilities=source_validation_probabilities,
        source_validation_labels=source_validation_labels,
        classes=[0, 1],
        max_iter=50,
    )

    assert result.method == "bbse_em"
    assert result.confusion_matrix is not None
    assert result.n_iterations >= 1
    assert result.target_prior[1] > result.target_prior[0]


def test_adjust_probabilities_to_prior_matches_requested_prior_direction() -> None:
    probabilities = np.asarray([[0.5, 0.5], [0.5, 0.5]])

    adjusted = adjust_probabilities_to_prior(probabilities, source_prior=[0.5, 0.5], target_prior=[0.8, 0.2])

    assert np.allclose(adjusted[0], [0.8, 0.2])
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_standalone_em_and_bbse_estimators() -> None:
    target_probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    em_prior, adjusted, iterations, converged = estimate_target_prior_em(
        target_probabilities,
        source_prior=[0.5, 0.5],
        max_iter=100,
    )
    assert em_prior.shape == (2,)
    assert adjusted.shape == target_probabilities.shape
    assert iterations >= 1
    assert isinstance(converged, bool)

    bbse_prior = estimate_target_prior_bbse(target_probabilities, confusion_matrix=np.eye(2))
    assert np.allclose(bbse_prior, target_probabilities.mean(axis=0), atol=1e-5)


def test_soft_confusion_matrix_columns_are_normalized() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])
    labels = [("left", 1), ("left", 1), ("right", 2), ("right", 2)]

    confusion = soft_confusion_matrix(probabilities, labels, classes=[("left", 1), ("right", 2)])

    assert confusion.shape == (2, 2)
    assert np.allclose(confusion.sum(axis=0), 1.0)
    assert confusion[0, 0] > confusion[1, 0]
    assert confusion[1, 1] > confusion[0, 1]


def test_mapping_source_prior_and_method_aliases() -> None:
    result = adapt_label_shift_probabilities(
        [[0.9, 0.1], [0.8, 0.2]],
        method="saerens-em",
        source_prior={"yes": 0.25, "no": 0.75},
        classes=["yes", "no"],
    )

    assert normalize_label_shift_method("black-box-shift") == "bbse"
    assert result.method == "em"
    assert np.allclose(result.source_prior, (0.25, 0.75))


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        adapt_label_shift_probabilities(
            [[0.8, 0.2], [0.2, 0.8]],
            source_prior=[0.5, 0.5],
            target_labels=[0, 1],  # type: ignore[call-arg]
        )


def test_bbse_requires_source_validation_inputs() -> None:
    with pytest.raises(ValueError, match="BBSE requires"):
        adapt_label_shift_probabilities([[0.8, 0.2]], method="bbse", source_prior=[0.5, 0.5])
