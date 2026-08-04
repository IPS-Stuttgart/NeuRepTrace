from __future__ import annotations

import numpy as np

from neureptrace.decoding.class_scores import class_score_matrix


class PredictsUnknownClass:
    def predict(self, features):
        return np.asarray(["known", "unknown"][: len(features)])


def test_prediction_fallback_rejects_labels_outside_class_order() -> None:
    scores, classes = class_score_matrix(
        PredictsUnknownClass(),
        np.zeros((2, 1)),
        fallback_labels=["known", "other"],
        predict_fallback=True,
    )

    assert scores is None
    assert classes is None
