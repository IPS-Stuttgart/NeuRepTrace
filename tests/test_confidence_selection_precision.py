from __future__ import annotations

import numpy as np

from neureptrace.decoding.confidence import select_confident_rows


def test_confidence_selection_uses_full_precision_at_threshold_boundaries() -> None:
    probabilities = np.asarray([[0.50000001, 0.49999999]], dtype=np.float64)
    exact_confidence = probabilities[0, 0]
    exact_margin = probabilities[0, 0] - probabilities[0, 1]
    exact_entropy = -np.sum(probabilities[0] * np.log(probabilities[0])) / np.log(2.0)

    rounded_confidence = float(np.float32(exact_confidence))
    rounded_margin = float(np.float32(exact_margin))
    rounded_entropy = float(np.float32(exact_entropy))
    assert rounded_confidence < exact_confidence
    assert rounded_margin < exact_margin
    assert rounded_entropy > exact_entropy

    result = select_confident_rows(
        probabilities,
        min_confidence=(rounded_confidence + exact_confidence) / 2.0,
        min_margin=(rounded_margin + exact_margin) / 2.0,
        max_entropy=(rounded_entropy + exact_entropy) / 2.0,
    )

    assert result.accepted_mask.tolist() == [True]
    assert result.confidence.dtype == np.float64
    assert result.margin.dtype == np.float64
    assert result.entropy.dtype == np.float64
    assert result.confidence[0] > rounded_confidence
    assert result.margin[0] > rounded_margin
    assert result.entropy[0] < rounded_entropy
