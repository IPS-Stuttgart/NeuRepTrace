from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_prior import (
    SOURCE_PRIOR_CATEGORY,
    adjust_probabilities_to_source_prior,
    estimate_source_class_prior,
    normalize_target_prior,
    source_prior_config,
)


def test_estimate_source_class_prior_empirical_order() -> None:
    prior, classes = estimate_source_class_prior(["b", "a", "b", "b"])

    assert classes.tolist() == ["b", "a"]
    assert np.allclose(prior, np.asarray([0.75, 0.25]))


def test_uniform_prior_adjustment_reweights_probabilities() -> None:
    probabilities = np.asarray([[0.75, 0.25], [0.50, 0.50]], dtype=float)

    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=["major", "major", "major", "minor"],
        classes=["major", "minor"],
        config={"target_prior": "uniform"},
    )

    assert result.probabilities.shape == probabilities.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.probabilities[0, 1] > probabilities[0, 1]
    assert np.allclose(result.source_prior, np.asarray([0.75, 0.25]))
    assert np.allclose(result.target_prior, np.asarray([0.5, 0.5]))
    assert result.metadata["source_prior_protocol_category"] == SOURCE_PRIOR_CATEGORY
    assert result.metadata["source_prior_uses_source_labels"] is True
    assert result.metadata["source_prior_uses_heldout_features"] is False
    assert result.metadata["source_prior_uses_heldout_labels"] is False
    assert result.metadata["source_prior_valid_for_strict_source_only"] is True


def test_source_prior_accepts_rectangular_numpy_composite_values() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    source_labels = np.asarray([("left", 1), ("right", 2), ("left", 1), ("right", 2)], dtype=object)
    classes = np.asarray([("left", 1), ("right", 2)], dtype=object)

    prior, inferred_classes = estimate_source_class_prior(source_labels, classes=classes)
    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=source_labels,
        classes=classes,
        config={"target_prior": "source"},
    )

    assert inferred_classes.tolist() == [("left", 1), ("right", 2)]
    assert result.classes.tolist() == [("left", 1), ("right", 2)]
    assert np.allclose(prior, np.asarray([0.5, 0.5]))
    assert np.allclose(result.source_prior, np.asarray([0.5, 0.5]))
    assert np.allclose(result.probabilities, probabilities)
    assert result.metadata["source_prior_n_classes"] == 2


def test_source_prior_target_source_is_identity_after_normalization() -> None:
    probabilities = np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=float)

    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=[0, 0, 1],
        classes=[0, 1],
        config={"target_prior": "source"},
    )

    assert np.allclose(result.probabilities, probabilities)


def test_source_prior_smoothing_and_aliases() -> None:
    assert normalize_target_prior("balanced") == "uniform"
    assert normalize_target_prior("empirical") == "source"
    cfg = source_prior_config(target_prior="source-prior", smoothing="1.0")
    assert cfg.target_prior == "source"
    assert cfg.smoothing == 1.0

    prior, _classes = estimate_source_class_prior(["a", "a", "b"], classes=["a", "b"], smoothing=1.0)
    assert np.allclose(prior, np.asarray([0.6, 0.4]))


@pytest.mark.parametrize("smoothing", [True, False, np.bool_(True), np.bool_(False), np.asarray(True), np.asarray(False)])
def test_source_prior_rejects_boolean_smoothing(smoothing) -> None:
    with pytest.raises(ValueError, match="smoothing must be non-negative and finite"):
        source_prior_config(smoothing=smoothing)


@pytest.mark.parametrize("epsilon", [True, False, np.bool_(True), np.bool_(False), np.asarray(True), np.asarray(False)])
def test_source_prior_rejects_boolean_epsilon(epsilon) -> None:
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        source_prior_config(epsilon=epsilon)


def test_source_prior_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="target_prior"):
        normalize_target_prior("bad")

    with pytest.raises(ValueError, match="absent from classes"):
        estimate_source_class_prior(["a", "b"], classes=["a"])

    with pytest.raises(ValueError, match="shape"):
        adjust_probabilities_to_source_prior([[0.5, 0.5, 0.0]], source_labels=[0, 1], classes=[0, 1])


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        adjust_probabilities_to_source_prior(
            [[0.5, 0.5]],
            source_labels=[0, 1],
            heldout_labels=[0],  # type: ignore[call-arg]
        )
