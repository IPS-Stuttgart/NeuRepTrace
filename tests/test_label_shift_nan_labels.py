from __future__ import annotations

import numpy as np

from neureptrace.decoding.label_shift import adapt_label_shift_probabilities, soft_confusion_matrix


def test_label_shift_source_labels_match_explicit_nan_class() -> None:
    result = adapt_label_shift_probabilities(
        [[0.7, 0.3], [0.3, 0.7]],
        source_labels=[np.nan, "seen", np.nan],
        classes=[np.nan, "seen"],
        max_iter=2,
    )

    assert len(result.classes) == 2
    assert np.isnan(result.classes[0])
    assert result.classes[1] == "seen"
    np.testing.assert_allclose(result.source_prior, (2.0 / 3.0, 1.0 / 3.0))
    assert result.metadata["label_shift_uses_source_labels"] is True


def test_label_shift_collapses_repeated_nan_labels_when_inferring_classes() -> None:
    result = adapt_label_shift_probabilities(
        [[0.7, 0.3], [0.3, 0.7]],
        source_labels=[np.nan, np.float64(np.nan), "seen"],
        max_iter=2,
    )

    assert len(result.classes) == 2
    assert np.isnan(result.classes[0])
    assert result.classes[1] == "seen"
    np.testing.assert_allclose(result.source_prior, (2.0 / 3.0, 1.0 / 3.0))


def test_label_shift_soft_confusion_accepts_nan_validation_class() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=float,
    )

    confusion = soft_confusion_matrix(
        probabilities,
        [np.nan, np.float64(np.nan), "seen", "seen"],
        classes=[np.nan, "seen"],
    )

    assert confusion.shape == (2, 2)
    np.testing.assert_allclose(confusion.sum(axis=0), 1.0)
    assert confusion[0, 0] > confusion[1, 0]
    assert confusion[1, 1] > confusion[0, 1]
