from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.selective_prediction import (
    SELECTIVE_PREDICTION_CATEGORY_COVERAGE,
    SELECTIVE_PREDICTION_CATEGORY_FIXED,
    confidence_threshold_for_coverage,
    probability_entropy,
    probability_margin,
    selective_predict,
)


def test_fixed_confidence_threshold_selects_high_confidence_rows() -> None:
    result = selective_predict(
        [[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]],
        classes=["left", "right"],
        confidence_threshold=0.75,
    )

    assert result.predictions.tolist() == ["left", "left", "right"]
    assert result.selected_mask.tolist() == [True, False, True]
    assert result.coverage == pytest.approx(2 / 3)
    assert result.metadata["selective_prediction_protocol_category"] == SELECTIVE_PREDICTION_CATEGORY_FIXED
    assert result.metadata["selective_prediction_uses_labels"] is False
    assert result.metadata["selective_prediction_adaptive_threshold"] is False


def test_target_coverage_sets_unlabeled_adaptive_threshold() -> None:
    result = selective_predict(
        [[0.95, 0.05], [0.80, 0.20], [0.65, 0.35], [0.51, 0.49]],
        target_coverage=0.5,
    )

    assert result.threshold == pytest.approx(0.8)
    assert result.selected_mask.tolist() == [True, True, False, False]
    assert result.metadata["selective_prediction_protocol_category"] == SELECTIVE_PREDICTION_CATEGORY_COVERAGE
    assert result.metadata["selective_prediction_adaptive_threshold"] is True
    assert result.metadata["selective_prediction_target_coverage"] == pytest.approx(0.5)


def test_entropy_and_margin_filters_are_combined() -> None:
    result = selective_predict(
        [[0.98, 0.02], [0.60, 0.40], [0.50, 0.50]],
        max_entropy=0.68,
        min_margin=0.3,
    )

    assert result.selected_mask.tolist() == [True, False, False]
    assert result.margin.tolist() == pytest.approx([0.96, 0.20, 0.0])
    assert np.all(result.entropy >= 0.0)


def test_probability_rows_are_normalized_defensively() -> None:
    result = selective_predict([[9.0, 1.0], [2.0, 2.0]])

    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.predictions.tolist() == [0, 0]
    assert result.confidence.tolist() == pytest.approx([0.9, 0.5])


def test_entropy_and_margin_helpers() -> None:
    probabilities = [[0.8, 0.2], [0.5, 0.5]]

    assert probability_margin(probabilities).tolist() == pytest.approx([0.6, 0.0])
    entropy = probability_entropy(probabilities)
    assert entropy[0] < entropy[1]


def test_confidence_threshold_for_coverage() -> None:
    threshold = confidence_threshold_for_coverage([0.2, 0.9, 0.7, 0.5], target_coverage=0.5)

    assert threshold == pytest.approx(0.7)


def test_classes_must_match_probability_columns() -> None:
    with pytest.raises(ValueError, match="classes"):
        selective_predict([[0.2, 0.8]], classes=["only_one"])


def test_invalid_probability_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        selective_predict([[0.8, -0.2]])


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"epsilon": np.asarray(1e-12)}, "epsilon"),
        ({"confidence_threshold": np.asarray([0.75])}, "confidence_threshold"),
        ({"max_entropy": np.asarray(0.68)}, "max_entropy"),
        ({"min_margin": np.asarray([0.3])}, "min_margin"),
        ({"target_coverage": np.asarray([0.5])}, "target_coverage"),
    ],
)
def test_array_valued_scalar_controls_are_rejected(kwargs: dict[str, object], name: str) -> None:
    with pytest.raises(ValueError, match=name):
        selective_predict([[0.9, 0.1], [0.2, 0.8]], **kwargs)  # type: ignore[arg-type]


def test_probability_entropy_rejects_array_valued_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        probability_entropy([[0.5, 0.5]], epsilon=np.asarray([1e-12]))  # type: ignore[arg-type]


def test_numpy_scalar_controls_remain_allowed() -> None:
    result = selective_predict(
        [[0.9, 0.1], [0.6, 0.4]],
        confidence_threshold=np.float64(0.5),
        max_entropy=np.float64(1.0),
        min_margin=np.float64(0.1),
        epsilon=np.float64(1e-12),
    )

    assert result.selected_mask.tolist() == [True, True]
    assert result.metadata["selective_prediction_confidence_threshold"] == pytest.approx(0.5)
    assert result.metadata["selective_prediction_max_entropy"] == pytest.approx(1.0)
    assert result.metadata["selective_prediction_min_margin"] == pytest.approx(0.1)


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        selective_predict(
            [[0.9, 0.1], [0.1, 0.9]],
            target_labels=[0, 1],  # type: ignore[call-arg]
        )
