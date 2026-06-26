from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_free import fit_source_free_predict_proba


class _SourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        probabilities = np.tile(np.array([[0.70, 0.30]], dtype=float), (x.shape[0], 1))
        probabilities[x[:, 0] > 0.0] = np.array([0.20, 0.80], dtype=float)
        return probabilities


def test_source_free_numeric_string_aliases_survive_metadata():
    target_features = np.array([[-1.0, 0.0], [2.0, 0.0]], dtype=float)

    result = fit_source_free_predict_proba(
        source_model=_SourceModel(),
        target_features=target_features,
        confidence_threshold="0.25",
        max_iterations="0.0",
        min_class_count="1.0",
        min_active_classes="1.0",
        prototype_weight="0.0",
        prototype_temperature="2.0",
        standardize_target="false",
    )

    assert result.probabilities.shape == (target_features.shape[0], 2)
    assert result.metadata["source_free_confidence_threshold"] == 0.25
    assert result.metadata["source_free_max_iterations"] == 0
    assert result.metadata["source_free_min_class_count"] == 1
    assert result.metadata["source_free_min_active_classes"] == 1
    assert result.metadata["source_free_prototype_weight"] == 0.0
    assert result.metadata["source_free_prototype_temperature"] == 2.0
    assert result.metadata["source_free_standardize_target"] is False
    assert result.adapter.max_iterations == 0
