from __future__ import annotations

import numpy as np

from neureptrace.decoding import PLSDiscriminantTransformer


def test_pls_da_preserves_rectangular_tuple_labels() -> None:
    labels = np.asarray(
        [
            ("face", "left"),
            ("house", "right"),
            ("face", "left"),
            ("house", "right"),
        ],
        dtype=object,
    )
    features = np.asarray(
        [
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.2],
            [1.1, 0.1, 0.0],
            [0.1, 1.1, 0.3],
        ],
        dtype=float,
    )

    transformer = PLSDiscriminantTransformer(n_components=1).fit(features, labels)
    transformed = transformer.transform(features)

    assert transformer.classes_.tolist() == [("face", "left"), ("house", "right")]
    assert transformer.n_components_ == 1
    assert transformed.shape == (4, 1)
