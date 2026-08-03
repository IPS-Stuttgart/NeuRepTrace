from __future__ import annotations

import numpy as np

from neureptrace.decoding.transfer import append_null_class_features


def test_append_null_class_features_preserves_single_composite_label() -> None:
    features, labels = append_null_class_features(
        np.asarray([[1.0]]),
        [("face", "left")],
        np.asarray([[0.0]]),
        null_label=("null", "baseline"),
    )

    assert features.tolist() == [[1.0], [0.0]]
    assert labels.dtype == object
    assert labels.tolist() == [("face", "left"), ("null", "baseline")]


def test_append_null_class_features_keeps_scalar_label_row_vectors() -> None:
    _features, labels = append_null_class_features(
        np.asarray([[1.0], [2.0]]),
        np.asarray([["left", "right"]], dtype=object),
    )

    assert labels.tolist() == ["left", "right"]
