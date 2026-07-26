from __future__ import annotations

import pytest

from neureptrace.decoding.selective_prediction import selective_predict


def test_target_coverage_keeps_exact_row_count_when_cutoff_confidences_tie() -> None:
    result = selective_predict(
        [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4]],
        target_coverage=0.5,
    )

    assert result.threshold == pytest.approx(0.6)
    assert result.selected_mask.tolist() == [True, True, False, False]
    assert result.coverage == pytest.approx(0.5)
    assert result.metadata["selective_prediction_selected_count"] == 2


def test_target_coverage_rounds_up_before_other_abstention_filters() -> None:
    result = selective_predict(
        [[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]],
        target_coverage=0.75,
        min_margin=0.5,
    )

    assert result.threshold == pytest.approx(0.6)
    assert result.selected_mask.tolist() == [True, True, False, False]
    assert result.coverage == pytest.approx(0.5)
