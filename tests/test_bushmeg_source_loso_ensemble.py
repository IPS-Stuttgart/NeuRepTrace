from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from neureptrace.bushmeg_source_loso_ensemble import (
    _apply_topk_pairwise_reranker,
    _fit_stacking_weights,
    _fit_topk_pairwise_reranker,
    _normalize_ensemble_class_bias,
    _normalize_rerank_top_k,
    _normalize_weighting,
    _parse_float_grid,
    _renormalize,
)


def test_stacked_weighting_is_configurable():
    assert _normalize_weighting("stacked") == "stacked"
    assert _normalize_ensemble_class_bias("balanced-acc") == "balanced_accuracy"


def test_fit_stacking_weights_prefers_source_oof_winner():
    labels = np.array([0, 1, 0, 1, 0, 1])
    good = np.array(
        [
            [0.90, 0.10],
            [0.15, 0.85],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.75, 0.25],
            [0.25, 0.75],
        ]
    )
    bad = 1.0 - good
    cube = np.stack([good, bad], axis=0)

    weights = _fit_stacking_weights(cube, labels, n_classes=2, max_iter=200)

    assert weights.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > 0.95
    assert weights[1] < 0.05


def test_fit_stacking_weights_rejects_invalid_probability_cube():
    labels = np.array([0, 1])
    cube = np.array(
        [
            [[0.8, 0.2], [0.1, 0.9]],
            [[0.4, 0.4], [0.2, 0.8]],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _fit_stacking_weights(cube, labels, n_classes=2, max_iter=20)


def test_parse_reranker_config_aliases():
    assert _normalize_rerank_top_k("off") == 0
    assert _normalize_rerank_top_k(3) == 3
    assert _parse_float_grid("0,0.5,1.0", [0.0]) == [0.0, 0.5, 1.0]


def test_topk_pairwise_reranker_can_flip_consistent_source_pair_confusion():
    probabilities = np.array(
        [
            [0.45, 0.50, 0.05],
            [0.40, 0.55, 0.05],
            [0.55, 0.40, 0.05],
            [0.50, 0.45, 0.05],
            [0.05, 0.10, 0.85],
            [0.10, 0.05, 0.85],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    baseline = balanced_accuracy_score(labels, probabilities.argmax(axis=1))

    reranker = _fit_topk_pairwise_reranker(
        probabilities,
        labels,
        n_classes=3,
        top_k=2,
        alpha_grid=[0.0, 1.0, 2.0, 4.0],
    )

    assert reranker is not None
    adjusted = _apply_topk_pairwise_reranker(probabilities, reranker)
    assert reranker.alpha > 0.0
    assert balanced_accuracy_score(labels, adjusted.argmax(axis=1)) > baseline


def test_topk_pairwise_reranker_rejects_invalid_fit_probabilities():
    probabilities = np.array([[0.6, 0.2], [0.1, 0.9]], dtype=float)
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _fit_topk_pairwise_reranker(probabilities, labels, n_classes=2, top_k=2)


def test_topk_pairwise_reranker_rejects_invalid_apply_probabilities():
    probabilities = np.array([[0.6, 0.2], [0.1, 0.9]], dtype=float)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _apply_topk_pairwise_reranker(probabilities, None)


def test_renormalize_rejects_invalid_ensemble_probabilities():
    probabilities = np.array([[0.5, 0.7], [0.1, 0.9]], dtype=float)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _renormalize(probabilities)
