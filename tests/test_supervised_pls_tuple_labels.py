from __future__ import annotations

import numpy as np

from neureptrace.bushmeg_supervised_lowrank_loso import SupervisedPLSTransformer


def test_supervised_pls_transformer_treats_tuple_labels_as_single_classes() -> None:
    rng = np.random.default_rng(29)
    features = rng.normal(size=(6, 4))
    labels = [
        ("face", "left"),
        ("object", "right"),
        ("scene", "left"),
        ("face", "left"),
        ("object", "right"),
        ("scene", "left"),
    ]

    transformer = SupervisedPLSTransformer(n_components=2).fit(features, labels)
    projected = transformer.transform(features)

    assert transformer.classes_.tolist() == [("face", "left"), ("object", "right"), ("scene", "left")]
    assert transformer.n_components_ == 2
    assert projected.shape == (6, 2)
