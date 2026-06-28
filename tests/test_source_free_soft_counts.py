from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_free import SourceFreeSubjectAdapter


class UnevenModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        probabilities = np.tile(np.array([[0.99, 0.01]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.51, 0.49], dtype=float)
        return probabilities


def test_soft_all_counts_use_effective_mass_not_raw_rows():
    x_target = np.vstack([np.array([[1.0, 0.0]], dtype=float), np.full((9, 2), -1.0)])
    model = UnevenModel()
    initial = model.predict_proba(x_target)

    adapter = SourceFreeSubjectAdapter(
        source_model=model,
        confidence_threshold=0.0,
        max_iterations=1,
        min_class_count=2,
        min_active_classes=1,
        prototype_weight=0.5,
        prototype_estimator="soft_all",
    ).fit(x_target)

    expected = []
    for class_index in range(initial.shape[1]):
        weights = initial[:, class_index]
        mass = float(np.sum(weights))
        expected.append(int(np.floor((mass * mass) / float(np.sum(weights * weights)) + 1.0e-12)))

    assert adapter.prototype_class_counts_.tolist() == expected
    assert adapter.prototype_class_counts_[1] < x_target.shape[0]
    assert adapter.active_classes_.tolist() == [True, False]
