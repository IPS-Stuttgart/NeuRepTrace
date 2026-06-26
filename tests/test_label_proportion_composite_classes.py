import numpy as np

from neureptrace.decoding.label_proportions import (
    adjust_probabilities_to_label_proportions,
    normalize_label_proportions,
    predict_labels_from_label_proportions,
)


def test_normalize_label_proportions_preserves_matrix_composite_classes():
    classes = np.asarray([[0, "left"], [1, "right"]], dtype=object)

    proportions, class_order = normalize_label_proportions([1, 3], classes=classes)

    assert class_order == ((0, "left"), (1, "right"))
    assert np.allclose(proportions, [0.25, 0.75])


def test_adjust_probabilities_to_label_proportions_preserves_matrix_composite_classes():
    classes = np.asarray([[0, "left"], [1, "right"]], dtype=object)

    result = adjust_probabilities_to_label_proportions(
        [[0.99, 0.01], [0.01, 0.99]],
        {(0, "left"): 1, (1, "right"): 1},
        classes=classes,
        tol=1e-12,
    )

    assert result.classes == ((0, "left"), (1, "right"))
    assert np.allclose(result.probabilities.mean(axis=0), [0.5, 0.5], atol=1e-8)
    assert list(predict_labels_from_label_proportions(result)) == [(0, "left"), (1, "right")]
