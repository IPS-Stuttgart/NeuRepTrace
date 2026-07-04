from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prototype_features import class_prototypes


def test_class_prototypes_preserve_explicit_composite_class_order() -> None:
    source = np.asarray([[0.0], [2.0], [10.0], [12.0]], dtype=float)
    labels = [("a", 1), ("a", 1), ("b", 2), ("b", 2)]

    prototypes = class_prototypes(source, labels, classes=[("b", 2), ("a", 1)])

    assert np.allclose(prototypes.ravel(), np.asarray([11.0, 1.0]))
