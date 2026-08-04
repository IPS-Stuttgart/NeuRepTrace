from __future__ import annotations

import numpy as np

from neureptrace.metrics.ranking import rank_class_scores


def _nested_labels(labels):
    return ((part for part in label) for label in labels)


def test_rank_class_scores_materializes_nested_generator_labels() -> None:
    result = rank_class_scores(
        np.asarray([[0.1, 0.9], [0.8, 0.2]]),
        _nested_labels([("run1", "cat"), ("run1", "dog")]),
        _nested_labels([("run1", "dog"), ("run1", "cat")]),
        top_k=(1,),
        row_top_k=1,
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.asarray([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: 1.0}
    assert result["rows"][0]["rank1_class"] == ("run1", "dog")
    assert result["rows"][0]["true_label_score"] == 0.9


def test_rank_class_scores_materializes_generator_labels_in_object_arrays() -> None:
    classes = np.empty(2, dtype=object)
    classes[:] = [(part for part in label) for label in [("run1", "cat"), ("run1", "dog")]]
    y_true = np.empty(1, dtype=object)
    y_true[:] = [(part for part in ("run1", "dog"))]

    result = rank_class_scores(
        np.asarray([[0.1, 0.9]]),
        classes,
        y_true,
        top_k=(1,),
        row_top_k=1,
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.asarray([1.0]))
    assert result["rows"][0]["rank1_class"] == ("run1", "dog")
    assert result["rows"][0]["true_label_score"] == 0.9
