from __future__ import annotations

import numpy as np

from neureptrace.bushmeg_source_loso_ensemble import TopKPairwiseReranker, _apply_topk_pairwise_reranker


def test_topk_pairwise_reranker_applies_pairwise_logit_once():
    probabilities = np.array([[0.25, 0.75]], dtype=float)
    reranker = TopKPairwiseReranker(
        n_classes=2,
        top_k=2,
        alpha=1.0,
        intercepts=(0.0, 0.0, 0.0, 0.0),
        slopes=(0.0, 1.0, 0.0, 0.0),
    )

    adjusted = _apply_topk_pairwise_reranker(probabilities, reranker)

    log_probabilities = np.log(probabilities)
    margin = log_probabilities[0, 0] - log_probabilities[0, 1]
    expected_scores = log_probabilities.copy()
    expected_scores[0, 0] += 0.5 * margin
    expected_scores[0, 1] -= 0.5 * margin
    expected = np.exp(expected_scores - expected_scores.max(axis=1, keepdims=True))
    expected = expected / expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(adjusted, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adjusted, [[0.1, 0.9]], rtol=1e-12, atol=1e-12)
