from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.confidence_selection import (
    CONFIDENCE_SELECTION_CATEGORY,
    confidence_selection_config,
    normalize_selection_mode,
    select_confident_probability_rows,
)


def test_threshold_selection_marks_confident_rows() -> None:
    probabilities = np.asarray(
        [
            [0.95, 0.05],
            [0.55, 0.45],
            [0.10, 0.90],
        ]
    )

    result = select_confident_probability_rows(probabilities, config={"threshold": 0.8})

    assert result.selected_mask.tolist() == [True, False, True]
    assert result.predicted_indices.tolist() == [0, 0, 1]
    assert result.selected_indices.tolist() == [0, 2]
    assert result.n_selected == 2
    assert result.metadata["confidence_selection_protocol_category"] == CONFIDENCE_SELECTION_CATEGORY
    assert result.metadata["confidence_selection_uses_true_labels"] is False
    assert result.metadata["confidence_selection_valid_for_unlabeled_target_adaptation"] is True


def test_top_k_selection_keeps_highest_confidence_rows() -> None:
    probabilities = np.asarray(
        [
            [0.70, 0.30],
            [0.99, 0.01],
            [0.60, 0.40],
            [0.20, 0.80],
        ]
    )

    result = select_confident_probability_rows(probabilities, config={"mode": "top_k", "top_k": 2})

    assert result.selected_indices.tolist() == [1, 3]
    assert result.selected_mask.tolist() == [False, True, False, True]


def test_per_class_top_k_keeps_one_row_per_predicted_class() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.10, 0.00],
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.65, 0.25],
            [0.10, 0.20, 0.70],
        ]
    )

    result = select_confident_probability_rows(
        probabilities,
        config={"mode": "per_class_top_k", "per_class_top_k": 1},
    )

    assert result.selected_indices.tolist() == [0, 2, 4]
    assert result.predicted_indices[result.selected_mask].tolist() == [0, 1, 2]


def test_margin_filter_combines_with_threshold() -> None:
    probabilities = np.asarray(
        [
            [0.80, 0.20, 0.00],
            [0.80, 0.19, 0.01],
            [0.60, 0.40, 0.00],
        ]
    )

    result = select_confident_probability_rows(
        probabilities,
        config={"threshold": 0.7, "min_margin": 0.5},
    )

    assert result.selected_mask.tolist() == [True, True, False]
    assert np.all(result.margins[result.selected_mask] >= 0.5)


def test_rows_are_normalized_before_selection() -> None:
    result = select_confident_probability_rows([[9.0, 1.0], [2.0, 8.0]], config={"threshold": 0.85})

    assert np.allclose(result.confidences, [0.9, 0.8])
    assert result.selected_mask.tolist() == [True, False]


def test_config_aliases_and_validation() -> None:
    assert normalize_selection_mode("topk") == "top_k"
    assert normalize_selection_mode("per-class") == "per_class_top_k"
    cfg = confidence_selection_config(mode="confidence", threshold="0.7", top_k="3")
    assert cfg.mode == "threshold"
    assert cfg.threshold == 0.7
    assert cfg.top_k == 3

    with pytest.raises(ValueError, match="Unknown confidence selection mode"):
        normalize_selection_mode("bad")


def test_optional_count_sentinels_are_normalized() -> None:
    assert confidence_selection_config(mode="top_k", top_k=" ALL ").top_k is None
    assert confidence_selection_config(mode="top_k", top_k="Null").top_k is None
    assert confidence_selection_config(mode="per_class_top_k", per_class_top_k=" full ").per_class_top_k is None


@pytest.mark.parametrize("epsilon", [0.0, "0", 1.0, "1", 2.0, "2"])
def test_epsilon_must_be_in_open_unit_interval(epsilon: float | str) -> None:
    with pytest.raises(ValueError, match=r"epsilon must be in \(0, 1\)"):
        confidence_selection_config(epsilon=epsilon)


def test_small_positive_epsilon_is_accepted() -> None:
    cfg = confidence_selection_config(epsilon="1e-6")

    assert cfg.epsilon == 1e-6


def test_large_epsilon_is_rejected_before_probability_clipping() -> None:
    with pytest.raises(ValueError, match=r"epsilon must be in \(0, 1\)"):
        select_confident_probability_rows([[0.99, 0.01]], config={"epsilon": 1.0})


def test_bad_probability_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        select_confident_probability_rows([[0.5, -0.5]])

    with pytest.raises(ValueError, match="positive total mass"):
        select_confident_probability_rows([[0.0, 0.0]])


def test_true_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        select_confident_probability_rows(
            [[0.9, 0.1]],
            true_labels=[0],  # type: ignore[call-arg]
        )
