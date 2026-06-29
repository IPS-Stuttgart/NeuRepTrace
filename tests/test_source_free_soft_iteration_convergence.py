from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_free import SourceFreeSubjectAdapter


class _CollapsedButMovingSoftModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.70, 0.30]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.60, 0.40], dtype=float)
        return probabilities


def test_soft_prototype_iterations_do_not_stop_on_unchanged_argmax_only():
    target_features = np.array([[-1.0], [1.0]], dtype=float)

    adapter = SourceFreeSubjectAdapter(
        source_model=_CollapsedButMovingSoftModel(),
        confidence_threshold=0.0,
        max_iterations=2,
        min_class_count=1,
        min_active_classes=2,
        prototype_weight=0.2,
        prototype_temperature=1.0,
        prototype_estimator="soft_all",
    ).fit(target_features)

    assert adapter.pseudo_labels_.tolist() == [0, 0]
    assert adapter.n_iterations_ == 2
    assert adapter.stop_reason_ == "max_iterations"
