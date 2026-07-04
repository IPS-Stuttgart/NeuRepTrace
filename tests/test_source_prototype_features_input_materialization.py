from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prototype_features import (
    class_prototypes,
    fit_source_prototype_features,
    prototype_distance_features,
)


def test_prototype_features_accept_one_pass_iterables() -> None:
    source_rows = ([value, 0.0] for value in [-2.0, -1.0, 1.0, 2.0])
    source_labels = (label for label in ["left", "left", "right", "right"])
    test_rows = ([value, 0.0] for value in [-1.5, 1.5])

    result = fit_source_prototype_features(
        source_features=source_rows,
        source_labels=source_labels,
        test_features=test_rows,
        config={"metric": "squared_euclidean", "use_diagonal_scale": False},
    )

    assert result.classes.tolist() == ["left", "right"]
    assert np.allclose(result.prototypes, np.asarray([[-1.5, 0.0], [1.5, 0.0]]))
    assert result.train_features.shape == (4, 2)
    assert result.test_features.shape == (2, 2)


def test_prototype_distance_features_accept_one_pass_iterables() -> None:
    features = ([value, 0.0] for value in [0.0, 2.0])
    prototypes = ([value, 0.0] for value in [0.0, 2.0])

    distances = prototype_distance_features(features, prototypes, feature_scale=[1.0, 1.0])

    assert np.allclose(np.diag(distances), 0.0)
    assert distances[0, 1] > distances[0, 0]
    assert distances[1, 0] > distances[1, 1]


def test_class_prototypes_preserve_explicit_composite_class_order() -> None:
    source = np.asarray([[0.0], [2.0], [10.0], [12.0]], dtype=float)
    labels = [("a", 1), ("a", 1), ("b", 2), ("b", 2)]

    prototypes = class_prototypes(source, labels, classes=[("b", 2), ("a", 1)])

    assert np.allclose(prototypes.ravel(), np.asarray([11.0, 1.0]))
