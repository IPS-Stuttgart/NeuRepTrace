from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_confidence import (
    TARGET_CONFIDENCE_CATEGORY,
    normalized_entropy,
    probability_margin,
    normalize_weighting_mode,
    target_confidence_config,
    target_confidence_weights,
)


def test_target_confidence_weights_metadata_and_pseudo_labels() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6], [0.51, 0.49]], dtype=float)

    result = target_confidence_weights(probabilities, classes=["a", "b"], config={"confidence_threshold": 0.55, "min_keep": 1})

    assert result.pseudo_labels.tolist() == ["a", "b", "a"]
    assert result.keep_mask.tolist() == [True, True, False]
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["target_confidence_protocol_category"] == TARGET_CONFIDENCE_CATEGORY
    assert result.metadata["target_confidence_uses_target_labels"] is False
    assert result.metadata["target_confidence_valid_for_unlabeled_target_adaptation"] is True
    assert result.metadata["target_confidence_valid_for_strict_source_only"] is False


def test_min_keep_restores_highest_confidence_rows() -> None:
    result = target_confidence_weights(
        [[0.51, 0.49], [0.52, 0.48], [0.53, 0.47]],
        config={"confidence_threshold": 0.9, "min_keep": 2, "weighting": "mask"},
    )

    assert result.keep_mask.tolist() == [False, True, True]
    assert result.sample_weights.tolist() == [0.0, 1.0, 1.0]


def test_margin_and_entropy_helpers() -> None:
    probabilities = np.asarray([[0.7, 0.2, 0.1], [1.0, 0.0, 0.0]], dtype=float)

    assert np.allclose(probability_margin(probabilities), np.asarray([0.5, 1.0]))
    entropy = normalized_entropy(probabilities)
    assert entropy.shape == (2,)
    assert entropy[0] > entropy[1]


def test_entropy_weighting_is_larger_for_low_entropy_rows() -> None:
    result = target_confidence_weights(
        [[0.5, 0.5], [0.95, 0.05]],
        config={"weighting": "entropy", "normalize_weights": False},
    )

    assert result.sample_weights[1] > result.sample_weights[0]


def test_aliases_and_validation() -> None:
    assert normalize_weighting_mode("prob") == "confidence"
    assert normalize_weighting_mode("low-entropy") == "entropy"
    assert target_confidence_config(min_keep="2", normalize_weights="false").normalize_weights is False

    with pytest.raises(ValueError, match="weighting"):
        normalize_weighting_mode("bad")

    with pytest.raises(ValueError, match="classes"):
        target_confidence_weights([[0.5, 0.5]], classes=["only_one"])


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        target_confidence_weights([[0.5, 0.5]], target_labels=[0])  # type: ignore[call-arg]
