from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_confidence_filter import filter_target_probabilities_by_confidence


def test_target_confidence_filter_preserves_matrix_composite_classes() -> None:
    probabilities = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    classes = np.asarray([["face", 1], ["tool", 2]], dtype=object)

    result = filter_target_probabilities_by_confidence(probabilities, classes=classes, config={"min_confidence": 0.0})

    assert result.pseudo_labels is not None
    assert result.selected_pseudo_labels is not None
    assert result.pseudo_labels.tolist() == [("face", 1), ("tool", 2)]
    assert result.selected_pseudo_labels.tolist() == [("face", 1), ("tool", 2)]


def test_target_confidence_filter_rejects_duplicate_composite_classes() -> None:
    probabilities = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    classes = np.asarray([["face", 1], ["face", 1]], dtype=object)

    with pytest.raises(ValueError, match="unique"):
        filter_target_probabilities_by_confidence(probabilities, classes=classes, config={"min_confidence": 0.0})
