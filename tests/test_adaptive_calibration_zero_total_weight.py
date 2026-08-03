import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from neureptrace._decoding_adaptive_calibration import AdaptiveCalibratedClassifierCV


def test_adaptive_calibration_rejects_zero_total_sample_weight() -> None:
    features = np.array([[-1.0], [0.0], [1.0]], dtype=float)
    labels = np.array([0, 0, 1])
    model = AdaptiveCalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=1000),
        method="sigmoid",
        cv=3,
    )

    with pytest.raises(ValueError, match="positive total weight"):
        model.fit(features, labels, sample_weight=np.zeros(labels.shape[0]))
