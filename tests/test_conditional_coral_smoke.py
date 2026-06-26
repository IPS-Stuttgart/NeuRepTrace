from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.conditional_coral import (
    CONDITIONAL_CORAL_CATEGORY,
    fit_conditional_coral_alignment,
    normalize_conditional_coral_fallback,
)


def test_conditional_coral_smoke() -> None:
    result = fit_conditional_coral_alignment(
        source_features=np.asarray([[0.0], [0.1], [2.0], [2.1]], dtype=float),
        source_labels=["class_a", "class_a", "class_b", "class_b"],
        target_features=np.asarray([[5.0], [5.1], [8.0], [8.1]], dtype=float),
        target_pseudo_labels=["class_a", "class_a", "class_b", "class_b"],
    )
    assert result.train_features.shape == (4, 1)
    assert result.test_features.shape == (4, 1)
    assert result.metadata["conditional_coral_protocol_category"] == CONDITIONAL_CORAL_CATEGORY
    assert result.metadata["conditional_coral_uses_target_labels"] is False


def test_conditional_coral_probability_input() -> None:
    result = fit_conditional_coral_alignment(
        source_features=np.asarray([[0.0], [0.1], [2.0], [2.1]], dtype=float),
        source_labels=["class_a", "class_a", "class_b", "class_b"],
        target_features=np.asarray([[5.0], [5.1], [8.0], [8.1]], dtype=float),
        target_probabilities=np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=float),
        classes=["class_a", "class_b"],
    )
    assert result.pseudo_labels.tolist() == ["class_a", "class_a", "class_b", "class_b"]
    assert np.allclose(result.pseudo_confidence, [0.9, 0.8, 0.8, 0.9])


def test_conditional_coral_fallback_validation() -> None:
    assert normalize_conditional_coral_fallback("global-coral") == "global"
    assert normalize_conditional_coral_fallback("off") == "identity"
    with pytest.raises(ValueError, match="fallback"):
        normalize_conditional_coral_fallback("invalid")
