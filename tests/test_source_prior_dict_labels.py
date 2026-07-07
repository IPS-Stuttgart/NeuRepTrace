from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior, estimate_source_class_prior


def test_source_prior_accepts_dict_labels_with_mixed_key_types() -> None:
    source_labels = [
        {"kind": "target", 1: "left"},
        {1: "left", "kind": "target"},
        {"kind": "other", 1: "right"},
    ]
    classes = [
        {1: "left", "kind": "target"},
        {"kind": "other", 1: "right"},
    ]

    prior, resolved_classes = estimate_source_class_prior(source_labels, classes=classes)
    result = adjust_probabilities_to_source_prior(
        [[0.7, 0.3], [0.4, 0.6]],
        source_labels=source_labels,
        classes=classes,
        config={"target_prior": "source"},
    )

    assert resolved_classes.shape == (2,)
    assert result.classes.shape == (2,)
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))
    np.testing.assert_allclose(result.source_prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))
    np.testing.assert_allclose(result.probabilities, np.asarray([[0.7, 0.3], [0.4, 0.6]]))
