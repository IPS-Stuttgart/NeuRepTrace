from __future__ import annotations

import numpy as np

from neureptrace.decoding.vrex import LinearVRExClassifier


def _parts(*values: str):
    return (value for value in values)


def test_vrex_canonicalizes_nested_generator_labels_and_domains() -> None:
    features = np.asarray(
        [
            [10.0, 2.0],
            [11.0, 2.5],
            [12.0, 3.0],
            [13.0, 3.5],
        ]
    )
    labels = [
        ("task", _parts("left", "hand")),
        ("task", _parts("right", "hand")),
        ("task", _parts("left", "hand")),
        ("task", _parts("right", "hand")),
    ]
    domains = [
        _parts("s1", "r1"),
        _parts("s1", "r1"),
        _parts("s2", "r1"),
        _parts("s2", "r1"),
    ]
    class_weight = {
        ("task", ("left", "hand")): 2.0,
        ("task", ("right", "hand")): 1.0,
    }

    model = LinearVRExClassifier(
        class_weight=class_weight,
        max_iter=3,
        tol=1e-4,
    )
    model.fit(features, labels, source_domains=domains)

    assert model.classes_.tolist() == [
        ("task", ("left", "hand")),
        ("task", ("right", "hand")),
    ]
    assert model.source_domains_.tolist() == [("s1", "r1"), ("s2", "r1")]
    assert model.class_weight_vector_.tolist() == [2.0, 1.0, 2.0, 1.0]
    assert model.metadata()["vrex_n_classes"] == 2
    assert model.metadata()["vrex_n_source_domains"] == 2


def test_vrex_canonicalizes_generator_identifiers_in_object_arrays() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    labels = np.empty(4, dtype=object)
    domains = np.empty(4, dtype=object)
    for index, (label, domain) in enumerate(
        [
            ("left", "s1"),
            ("right", "s1"),
            ("left", "s2"),
            ("right", "s2"),
        ]
    ):
        labels[index] = _parts("task", label)
        domains[index] = _parts(domain, "run1")

    model = LinearVRExClassifier(max_iter=3, tol=1e-4)
    model.fit(features, labels, source_domains=domains)

    assert model.classes_.tolist() == [("task", "left"), ("task", "right")]
    assert model.source_domains_.tolist() == [("s1", "run1"), ("s2", "run1")]
