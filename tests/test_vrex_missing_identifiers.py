from __future__ import annotations

import numpy as np

from neureptrace.decoding.vrex import LinearVRExClassifier


def test_vrex_groups_missing_labels_and_domains_with_mapping_weights() -> None:
    source_features = np.asarray(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    source_labels = np.empty(4, dtype=object)
    source_labels[:] = [float("nan"), "right", float("nan"), "right"]
    source_domains = np.empty(4, dtype=object)
    source_domains[:] = [float("nan"), float("nan"), "session-2", "session-2"]

    model = LinearVRExClassifier(
        class_weight={float("nan"): 2.0, "right": 1.0},
        max_iter=10,
        tol=1.0e-4,
    )
    model.fit(source_features, source_labels, source_domains=source_domains)

    assert model.n_classes_ == 2
    assert model.n_source_domains_ == 2
    assert np.isnan(model.classes_[0])
    assert model.classes_[1] == "right"
    assert np.isnan(model.source_domains_[0])
    assert model.source_domains_[1] == "session-2"
    np.testing.assert_allclose(model.class_weight_vector_, [2.0, 1.0, 2.0, 1.0])

    probabilities = model.predict_proba(source_features)
    assert probabilities.shape == (4, 2)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
