from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_free import SourceFreeSubjectAdapter


class _TupleLabelProbabilityModel:
    def __init__(self):
        self.classes_ = np.asarray([("house", "late"), ("face", "early")], dtype=object)

    def predict_proba(self, features):
        probabilities = np.asarray(
            [
                [0.10, 0.90],
                [0.80, 0.20],
                [0.45, 0.55],
            ],
            dtype=float,
        )
        return probabilities[: np.asarray(features).shape[0]]


def test_source_free_adapter_preserves_tuple_labels_and_aligns_columns():
    requested_classes = [("face", "early"), ("house", "late")]
    target_features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=float,
    )

    adapter = SourceFreeSubjectAdapter(source_model=_TupleLabelProbabilityModel(), max_iterations=0).fit(target_features, classes=requested_classes)

    assert adapter.classes_.shape == (2,)
    assert adapter.classes_.tolist() == requested_classes
    np.testing.assert_allclose(
        adapter.probabilities_,
        np.asarray(
            [
                [0.90, 0.10],
                [0.20, 0.80],
                [0.55, 0.45],
            ],
            dtype=float,
        ),
    )
    assert adapter.predict(target_features).tolist() == [requested_classes[0], requested_classes[1], requested_classes[0]]
