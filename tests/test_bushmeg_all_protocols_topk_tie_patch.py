from __future__ import annotations

import numpy as np
import pytest

from neureptrace._bushmeg_all_protocols_topk_tie_patch import _top_k_accuracy


def test_top_k_accuracy_rejects_invalid_k_values() -> None:
    probabilities = np.asarray([[0.6, 0.4], [0.3, 0.7]], dtype=float)
    labels = np.asarray([0, 1], dtype=int)

    for invalid_k in (0, -1, True, 1.5, float("nan")):
        with pytest.raises(ValueError, match="positive integer"):
            _top_k_accuracy(probabilities, labels, k=invalid_k)


def test_top_k_accuracy_preserves_exact_tie_ordering() -> None:
    probabilities = np.asarray([[0.5, 0.5, 0.1], [0.2, 0.4, 0.4]], dtype=float)
    labels = np.asarray([0, 2], dtype=int)

    assert _top_k_accuracy(probabilities, labels, k=1) == 0.5
    assert _top_k_accuracy(probabilities, labels, k=2) == 1.0


def test_top_k_accuracy_accepts_integral_string_and_column_labels() -> None:
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.1, 0.7]], dtype=float)

    assert _top_k_accuracy(probabilities, ["0", "2"], k=1) == 1.0
    assert _top_k_accuracy(probabilities, np.asarray([[0], [2]], dtype=float), k=1) == 1.0


def test_top_k_accuracy_rejects_fractional_label_indices() -> None:
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], dtype=float)

    with pytest.raises(ValueError, match="integer class indices"):
        _top_k_accuracy(probabilities, np.asarray([0.0, 1.5]), k=1)


def test_top_k_accuracy_rejects_boolean_label_indices() -> None:
    probabilities = np.asarray([[0.7, 0.2], [0.2, 0.7]], dtype=float)

    with pytest.raises(ValueError, match="integer class indices"):
        _top_k_accuracy(probabilities, np.asarray([False, True], dtype=object), k=1)


def test_top_k_accuracy_rejects_multicolumn_label_indices() -> None:
    probabilities = np.asarray([[0.7, 0.2], [0.2, 0.7]], dtype=float)

    with pytest.raises(ValueError, match="one entry per probability row"):
        _top_k_accuracy(probabilities, np.asarray([[0, 1]]), k=1)
