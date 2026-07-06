from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_oracle_prior import (
    ORACLE_TARGET_PRIOR_CATEGORY,
    apply_oracle_target_prior,
    oracle_target_prior,
)


def test_oracle_target_prior_uses_target_labels_and_marks_oracle() -> None:
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.4, 0.6],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "a", "b"], dtype=object)

    result = apply_oracle_target_prior(probabilities, labels, classes=["a", "b"], source_prior=[0.5, 0.5])

    assert result.probabilities.shape == probabilities.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.target_prior, np.asarray([0.75, 0.25], dtype=np.float32))
    assert np.allclose(result.source_prior, np.asarray([0.5, 0.5], dtype=np.float32))
    assert np.allclose(result.prior_ratio, np.asarray([1.5, 0.5], dtype=np.float32))
    assert result.metadata["oracle_target_prior_protocol_category"] == ORACLE_TARGET_PRIOR_CATEGORY
    assert result.metadata["oracle_target_prior_uses_target_labels"] is True
    assert result.metadata["oracle_target_prior_debug_upper_bound"] is True
    assert result.metadata["oracle_target_prior_valid_for_benchmark"] is False


def test_oracle_target_prior_function_respects_class_order() -> None:
    labels = ["b", "a", "b", "b"]

    prior = oracle_target_prior(labels, classes=["a", "b"])

    assert np.allclose(prior, np.asarray([0.25, 0.75]))


def test_oracle_target_prior_preserves_composite_labels() -> None:
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.6, 0.4],
        ],
        dtype=float,
    )
    labels = [("face", "left"), ("face", "left"), ("scene", "right"), ("face", "left")]
    classes = [("face", "left"), ("scene", "right")]

    result = apply_oracle_target_prior(probabilities, labels, classes=classes, source_prior=[0.5, 0.5])
    prior = oracle_target_prior(np.asarray(labels, dtype=object), classes=np.asarray(classes, dtype=object))

    assert result.classes.tolist() == classes
    assert np.allclose(result.target_prior, np.asarray([0.75, 0.25], dtype=np.float32))
    assert np.allclose(prior, np.asarray([0.75, 0.25]))


def test_oracle_target_prior_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="absent from classes"):
        oracle_target_prior(["a", "c"], classes=["a", "b"])


def test_oracle_target_prior_rejects_bad_source_prior_length() -> None:
    with pytest.raises(ValueError, match="source_prior"):
        apply_oracle_target_prior([[0.5, 0.5]], ["a"], classes=["a", "b"], source_prior=[1.0, 0.0, 0.0])


def test_oracle_target_prior_requires_label_per_probability_row() -> None:
    with pytest.raises(ValueError, match="one value per probability row"):
        apply_oracle_target_prior([[0.5, 0.5], [0.6, 0.4]], ["a"], classes=["a", "b"])
