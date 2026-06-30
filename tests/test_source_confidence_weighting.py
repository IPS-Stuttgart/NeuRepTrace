from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_confidence_weighting import (
    SOURCE_CONFIDENCE_WEIGHT_CATEGORY,
    compute_source_confidence_weights,
    confidence_scores,
    normalize_confidence_weight_mode,
    source_confidence_weight_config,
)


def test_confidence_weighting_uses_max_probability() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.55, 0.45], [0.5, 0.5]], dtype=float)

    result = compute_source_confidence_weights(probabilities, config={"mode": "confidence", "min_weight": 0.0})

    assert np.isclose(np.mean(result.sample_weights), 1.0)
    assert result.scores.tolist() == pytest.approx([0.9, 0.55, 0.5])
    assert result.sample_weights[0] > result.sample_weights[-1]
    assert result.metadata["source_confidence_weighting_protocol_category"] == SOURCE_CONFIDENCE_WEIGHT_CATEGORY
    assert result.metadata["source_confidence_weighting_uses_heldout_labels"] is False
    assert result.metadata["source_confidence_weighting_valid_for_strict_source_only"] is True


def test_correct_confidence_uses_source_labels() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.55, 0.45]], dtype=float)

    result = compute_source_confidence_weights(
        probabilities,
        source_labels=[0, 1],
        config={"mode": "correct_confidence", "min_weight": 0.0, "normalize_weights": False},
    )

    assert result.scores.tolist() == pytest.approx([0.9, 0.45])
    assert result.sample_weights.tolist() == pytest.approx([0.9, 0.45])
    assert result.metadata["source_confidence_weighting_uses_source_labels"] is True


def test_margin_and_entropy_scores() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.5, 0.5]], dtype=float)

    margin = confidence_scores(probabilities, mode="margin")
    entropy = confidence_scores(probabilities, mode="entropy")

    assert margin.tolist() == pytest.approx([0.6, 0.0])
    assert entropy[0] > entropy[1]
    assert entropy[1] == pytest.approx(0.0)


def test_source_confidence_weighting_rejects_zero_mass_rows() -> None:
    with pytest.raises(ValueError, match="positive probability mass"):
        compute_source_confidence_weights([[0.0, 0.0], [0.7, 0.3]])

    with pytest.raises(ValueError, match="positive probability mass"):
        confidence_scores([[0.0, 0.0]], mode="confidence")


def test_aliases_and_validation() -> None:
    assert normalize_confidence_weight_mode("max-prob") == "confidence"
    assert normalize_confidence_weight_mode("label-confidence") == "correct_confidence"
    assert normalize_confidence_weight_mode("low-entropy") == "entropy"
    cfg = source_confidence_weight_config(min_weight="0.2", normalize_weights="false")
    assert cfg.min_weight == 0.2
    assert cfg.normalize_weights is False

    with pytest.raises(ValueError, match="weighting mode"):
        normalize_confidence_weight_mode("bad")

    with pytest.raises(ValueError, match="labels are required"):
        confidence_scores([[0.5, 0.5]], mode="correct_confidence")

    with pytest.raises(ValueError, match="source_labels"):
        compute_source_confidence_weights([[0.5, 0.5]], source_labels=[2], config={"mode": "correct_confidence"})


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        compute_source_confidence_weights([[0.5, 0.5]], heldout_labels=[0])  # type: ignore[call-arg]
