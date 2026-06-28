from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.confidence import accepted_probability_rows, confidence_scores, select_confident_rows


def test_confidence_scores_return_expected_values() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.6], [2.0, 1.0]], dtype=float)

    confidence, margin, entropy, predicted = confidence_scores(probabilities)

    assert predicted.tolist() == [0, 1, 0]
    assert np.allclose(confidence, [0.8, 6 / 7, 2 / 3])
    assert np.allclose(margin, [0.6, 5 / 7, 1 / 3])
    assert np.all(entropy >= 0.0)
    assert np.all(entropy <= 1.0)


def test_confidence_scores_break_probability_ties_by_lowest_index() -> None:
    probabilities = np.asarray(
        [
            [0.5, 0.5, 0.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.8, 0.8],
        ],
        dtype=float,
    )

    confidence, margin, _entropy, predicted = confidence_scores(probabilities)

    assert predicted.tolist() == [0, 0, 1]
    assert np.allclose(confidence, [0.5, 1 / 3, 4 / 9])
    assert np.allclose(margin, [0.0, 0.0, 0.0])


def test_select_confident_rows_applies_all_thresholds() -> None:
    probabilities = np.asarray([[0.95, 0.05], [0.55, 0.45], [0.7, 0.3]], dtype=float)

    result = select_confident_rows(probabilities, min_confidence=0.7, min_margin=0.3, max_entropy=0.9)

    assert result.accepted_mask.tolist() == [True, False, True]
    assert result.metadata["confidence_selection_n_rows"] == 3
    assert result.metadata["confidence_selection_n_accepted"] == 2
    assert np.isclose(result.metadata["confidence_selection_acceptance_rate"], 2 / 3)


def test_accepted_probability_rows_returns_normalized_subset() -> None:
    probabilities = np.asarray([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]], dtype=float)
    selection = select_confident_rows(probabilities, min_margin=0.9)

    accepted = accepted_probability_rows(probabilities, selection=selection)

    assert accepted.shape == (2, 2)
    assert np.allclose(accepted.sum(axis=1), 1.0)
    assert np.allclose(accepted, [[1.0, 0.0], [0.0, 1.0]])


def test_confidence_guardrails() -> None:
    with pytest.raises(ValueError):
        confidence_scores([[1.0]])
    with pytest.raises(ValueError):
        confidence_scores([[1.0, -0.1]])
    with pytest.raises(ValueError):
        select_confident_rows([[0.5, 0.5]], min_confidence=1.5)


def test_selection_mask_must_match_rows() -> None:
    selection = select_confident_rows([[0.9, 0.1], [0.1, 0.9]])
    bad_selection = type(selection)(
        confidence=selection.confidence[:1],
        margin=selection.margin[:1],
        entropy=selection.entropy[:1],
        predicted_index=selection.predicted_index[:1],
        accepted_mask=selection.accepted_mask[:1],
        metadata=selection.metadata,
    )

    with pytest.raises(ValueError):
        accepted_probability_rows([[0.9, 0.1], [0.1, 0.9]], selection=bad_selection)
