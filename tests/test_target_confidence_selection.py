from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_confidence_selection import (
    TARGET_CONFIDENCE_SELECTION_CATEGORY,
    select_target_confident_predictions,
    target_confidence_selection_config,
)


def test_target_confidence_selection_keeps_thresholded_rows() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]], dtype=float)

    result = select_target_confident_predictions(
        probabilities,
        classes=["a", "b"],
        config={"min_confidence": 0.75, "min_margin": 0.5},
    )

    assert result.predictions.tolist() == ["a", "a", "b"]
    assert result.keep_mask.tolist() == [True, False, True]
    assert result.selected_indices.tolist() == [0, 2]
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["target_confidence_selection_protocol_category"] == TARGET_CONFIDENCE_SELECTION_CATEGORY
    assert result.metadata["target_confidence_selection_uses_target_probabilities"] is True
    assert result.metadata["target_confidence_selection_uses_target_labels"] is False
    assert result.metadata["target_confidence_selection_valid_for_unlabeled_target_adaptation"] is True


def test_top_fraction_selects_most_confident_rows() -> None:
    probabilities = np.asarray([[0.51, 0.49], [0.99, 0.01], [0.75, 0.25], [0.6, 0.4]], dtype=float)

    result = select_target_confident_predictions(probabilities, config={"top_fraction": 0.5})

    assert result.selected_indices.tolist() == [1, 2]
    assert result.metadata["target_confidence_selection_n_selected_rows"] == 2


def test_min_keep_rows_forces_confident_fallback() -> None:
    probabilities = np.asarray([[0.51, 0.49], [0.55, 0.45], [0.8, 0.2]], dtype=float)

    result = select_target_confident_predictions(probabilities, config={"min_confidence": 0.95, "min_keep_rows": 2})

    assert result.selected_indices.tolist() == [1, 2]
    assert result.keep_mask.tolist() == [False, True, True]


def test_classes_validation_and_config_validation() -> None:
    with pytest.raises(ValueError, match="classes"):
        select_target_confident_predictions([[0.5, 0.5]], classes=["only"])

    assert target_confidence_selection_config(min_keep_rows="2").min_keep_rows == 2

    with pytest.raises(ValueError, match="top_fraction"):
        target_confidence_selection_config(top_fraction=0.0)


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        select_target_confident_predictions([[0.5, 0.5]], target_labels=[0])  # type: ignore[call-arg]
