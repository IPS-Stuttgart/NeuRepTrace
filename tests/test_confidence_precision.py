from __future__ import annotations

import numpy as np

from neureptrace.decoding.confidence import accepted_probability_rows, confidence_scores, select_confident_rows


def test_confidence_entropy_preserves_tiny_positive_probability_mass() -> None:
    probabilities = np.asarray([[1.0, 1.0e-50]], dtype=float)

    confidence, margin, entropy, predicted = confidence_scores(probabilities)
    selection = select_confident_rows(probabilities, max_entropy=0.0)

    assert predicted.tolist() == [0]
    np.testing.assert_array_equal(confidence, [1.0])
    np.testing.assert_array_equal(margin, [1.0])
    assert entropy.dtype == np.float64
    assert entropy[0] > 0.0
    assert selection.accepted_mask.tolist() == [False]


def test_accepted_probability_rows_preserve_tiny_positive_probability_mass() -> None:
    probabilities = np.asarray([[1.0, 1.0e-50]], dtype=float)
    selection = select_confident_rows(probabilities)

    accepted = accepted_probability_rows(probabilities, selection=selection)

    assert accepted.dtype == np.float64
    assert accepted[0, 1] > 0.0
    np.testing.assert_array_equal(accepted, probabilities)


def test_confidence_thresholds_use_full_precision_before_compaction() -> None:
    probabilities = np.asarray([[0.50000001, 0.49999999]], dtype=float)

    selection = select_confident_rows(probabilities, min_confidence=0.50000002)

    assert selection.confidence.dtype == np.float32
    np.testing.assert_array_equal(selection.confidence, [0.5])
    assert selection.accepted_mask.tolist() == [False]
