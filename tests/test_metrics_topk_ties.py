import numpy as np
import pytest

from neureptrace.metrics import top_k_accuracy, weighted_top_k_accuracy


def test_top_k_resolves_boundary_ties_without_expanding_top_k_set():
    scores = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.1, 0.45, 0.45],
            [0.6, 0.2, 0.2],
        ]
    )
    labels = np.array([0, 1, 2])

    assert top_k_accuracy(scores, labels, k=1) == pytest.approx(2 / 3)
    assert top_k_accuracy(scores, labels, k=2) == pytest.approx(2 / 3)


def test_top_k_uniform_probabilities_match_deterministic_top_k_set_size():
    scores = np.full((3, 3), 1.0 / 3.0)
    labels = np.array([0, 1, 2])

    assert top_k_accuracy(scores, labels, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy(scores, labels, k=2) == pytest.approx(2 / 3)


def test_weighted_top_k_resolves_boundary_ties_without_expanding_top_k_set():
    scores = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.1, 0.45, 0.45],
            [0.6, 0.2, 0.2],
        ]
    )
    labels = np.array([0, 1, 2])
    weights = np.array([1.0, 2.0, 3.0])

    assert weighted_top_k_accuracy(scores, labels, weights, k=1) == pytest.approx(0.5)
    assert weighted_top_k_accuracy(scores, labels, weights, k=2) == pytest.approx(0.5)


def test_weighted_top_k_uniform_probabilities_match_deterministic_top_k_set_size():
    scores = np.full((3, 3), 1.0 / 3.0)
    labels = np.array([0, 1, 2])
    weights = np.array([1.0, 2.0, 3.0])

    assert weighted_top_k_accuracy(scores, labels, weights, k=1) == pytest.approx(1 / 6)
    assert weighted_top_k_accuracy(scores, labels, weights, k=2) == pytest.approx(0.5)
