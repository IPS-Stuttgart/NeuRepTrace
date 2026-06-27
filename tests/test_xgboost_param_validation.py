from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.classifiers import train_classifier


def test_xgboost_rejects_boolean_classifier_param() -> None:
    features = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=int)

    with pytest.raises(ValueError, match="xgboost classifier_param"):
        train_classifier(features, labels, "xgboost", True)
