from __future__ import annotations

import numpy as np

from neureptrace.decoding.confidence_selection import select_confident_probability_rows


def test_probability_ties_choose_lowest_class_index() -> None:
    probabilities = np.asarray(
        [
            [0.50, 0.50, 0.00],
            [1.0, 1.0, 1.0],
            [0.20, 0.40, 0.40],
        ]
    )

    result = select_confident_probability_rows(probabilities, config={"threshold": 0.0})

    assert result.predicted_indices.tolist() == [0, 0, 1]
    assert np.allclose(result.margins, [0.0, 0.0, 0.0])


def test_top_k_selection_breaks_row_ties_by_input_order() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.90, 0.10],
            [0.70, 0.30],
        ]
    )

    result = select_confident_probability_rows(probabilities, config={"mode": "top_k", "top_k": 1})

    assert result.selected_indices.tolist() == [0]


def test_per_class_top_k_selection_breaks_row_ties_by_input_order() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.90, 0.10],
            [0.20, 0.80],
            [0.20, 0.80],
        ]
    )

    result = select_confident_probability_rows(
        probabilities,
        config={"mode": "per_class_top_k", "per_class_top_k": 1},
    )

    assert result.selected_indices.tolist() == [0, 2]
    assert result.predicted_indices[result.selected_mask].tolist() == [0, 1]
