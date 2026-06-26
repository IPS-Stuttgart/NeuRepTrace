from __future__ import annotations

import numpy as np

from neureptrace.decoding.label_shift import adapt_label_shift_probabilities


def test_explicit_source_prior_metadata_does_not_claim_source_labels() -> None:
    result = adapt_label_shift_probabilities(
        np.asarray([[0.85, 0.15], [0.75, 0.25]], dtype=float),
        source_prior=[0.4, 0.6],
        classes=["left", "right"],
    )

    assert result.metadata["label_shift_uses_source_labels"] is False
    assert result.metadata["label_shift_uses_source_validation_probabilities"] is False
    assert result.metadata["label_shift_uses_target_probabilities"] is True
    assert result.metadata["label_shift_uses_target_labels"] is False


def test_inferred_source_prior_metadata_reports_source_labels() -> None:
    result = adapt_label_shift_probabilities(
        np.asarray([[0.85, 0.15], [0.15, 0.85]], dtype=float),
        source_labels=["left", "left", "right", "right"],
        classes=["left", "right"],
    )

    assert result.metadata["label_shift_uses_source_labels"] is True
