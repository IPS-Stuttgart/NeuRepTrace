from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_confidence_gate import (
    TARGET_CONFIDENCE_GATE_CATEGORY,
    gate_target_probabilities_by_confidence,
    normalize_score_mode,
    normalize_threshold_mode,
    target_confidence_gate_config,
    target_confidence_scores,
)


def test_fixed_max_probability_gate_returns_predictions_and_metadata() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]], dtype=float)

    result = gate_target_probabilities_by_confidence(
        probabilities,
        classes=["left", "right"],
        config={"score": "max_probability", "threshold_mode": "fixed", "confidence_threshold": 0.8},
    )

    assert result.predictions.tolist() == ["left", "left", "right"]
    assert result.accepted_mask.tolist() == [True, False, True]
    assert result.rejected_mask.tolist() == [False, True, False]
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["target_confidence_gate_protocol_category"] == TARGET_CONFIDENCE_GATE_CATEGORY
    assert result.metadata["target_confidence_gate_uses_target_probabilities"] is True
    assert result.metadata["target_confidence_gate_uses_target_labels"] is False
    assert result.metadata["target_confidence_gate_valid_for_unlabeled_target_adaptation"] is True
    assert result.metadata["target_confidence_gate_n_accepted"] == 2


def test_retain_fraction_threshold_accepts_top_scores() -> None:
    probabilities = np.asarray([[0.95, 0.05], [0.7, 0.3], [0.6, 0.4], [0.51, 0.49]], dtype=float)

    result = gate_target_probabilities_by_confidence(
        probabilities,
        config={"threshold_mode": "retain_fraction", "retain_fraction": 0.5},
    )

    assert result.accepted_mask.tolist() == [True, True, False, False]
    assert np.isclose(result.threshold, np.quantile(np.asarray([0.95, 0.7, 0.6, 0.51]), 0.5))


def test_margin_and_entropy_scores() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.5, 0.5]], dtype=float)

    margin = target_confidence_scores(probabilities, score="margin")
    confidence = target_confidence_scores(probabilities, score="normalized_confidence")

    assert np.allclose(margin, np.asarray([0.8, 0.0]))
    assert confidence[0] > confidence[1]
    assert np.isclose(confidence[1], 0.0)


def test_class_length_and_probability_validation() -> None:
    with pytest.raises(ValueError, match="classes"):
        gate_target_probabilities_by_confidence([[0.5, 0.5]], classes=["only_one"])

    with pytest.raises(ValueError, match="probabilities"):
        gate_target_probabilities_by_confidence([[0.5, -0.5]])


def test_aliases_and_config_validation() -> None:
    assert normalize_score_mode("top-gap") == "margin"
    assert normalize_score_mode("entropy") == "normalized_confidence"
    assert normalize_threshold_mode("quantile") == "retain_fraction"
    assert target_confidence_gate_config(retain_fraction="0.25").retain_fraction == 0.25

    with pytest.raises(ValueError, match="confidence score"):
        normalize_score_mode("bad")

    with pytest.raises(ValueError, match="retain_fraction"):
        target_confidence_gate_config(retain_fraction=0.0)


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        gate_target_probabilities_by_confidence(
            [[0.5, 0.5]],
            target_labels=[0],  # type: ignore[call-arg]
        )
