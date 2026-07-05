from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_confidence_filter import (
    TargetConfidenceFilterConfig,
    filter_target_probabilities_by_confidence,
    probability_entropy,
    target_confidence_filter_config,
)


def test_target_confidence_filter_selects_and_orders_pseudo_labels() -> None:
    probabilities = np.asarray([[0.90, 0.10], [0.55, 0.45], [0.20, 0.80]], dtype=float)

    result = filter_target_probabilities_by_confidence(
        probabilities,
        classes=["left", "right"],
        config={"min_confidence": 0.75, "sort_by": "confidence"},
    )

    assert result.selected_mask.tolist() == [True, False, True]
    assert result.selected_indices.tolist() == [0, 2]
    assert result.selected_pseudo_labels.tolist() == ["left", "right"]
    assert np.allclose(result.selected_probabilities.sum(axis=1), 1.0)
    assert result.metadata["target_confidence_filter_n_selected"] == 2
    assert result.metadata["target_confidence_filter_uses_target_labels"] is False


def test_target_confidence_filter_direct_config_normalizes_scalars() -> None:
    cfg = TargetConfidenceFilterConfig(
        min_confidence=np.asarray([0.5]),
        max_entropy=np.asarray("none"),
        top_k=np.asarray([1]),
        sort_by=np.asarray(["conf"]),
        epsilon=np.asarray(1e-6),
    )

    assert cfg.min_confidence == pytest.approx(0.5)
    assert cfg.max_entropy is None
    assert cfg.top_k == 1
    assert cfg.sort_by == "confidence"
    assert cfg.epsilon == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_confidence", True),
        ("min_confidence", np.asarray(True)),
        ("max_entropy", False),
        ("top_k", np.bool_(True)),
        ("top_k", np.asarray([False], dtype=object)),
        ("epsilon", np.asarray(False)),
    ],
)
def test_target_confidence_filter_config_rejects_boolean_numeric_controls(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        target_confidence_filter_config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_confidence", np.asarray([0.2, 0.3])),
        ("max_entropy", np.asarray([0.1, 0.2])),
        ("top_k", np.asarray([1, 2])),
        ("epsilon", np.asarray([1e-6, 1e-5])),
    ],
)
def test_target_confidence_filter_config_rejects_vector_numeric_controls(field: str, value: np.ndarray) -> None:
    with pytest.raises(ValueError, match=field):
        target_confidence_filter_config(**{field: value})


def test_target_confidence_filter_rejects_boolean_probability_rows() -> None:
    with pytest.raises(ValueError, match="booleans"):
        filter_target_probabilities_by_confidence(np.asarray([[True, False], [False, True]], dtype=bool))

    with pytest.raises(ValueError, match="booleans"):
        probability_entropy([[1.0, False]])
