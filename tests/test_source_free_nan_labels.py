from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free import fit_source_free_predict_proba


class _NanClassSourceModel:
    classes_ = np.array([np.nan, "stimulus"], dtype=object)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.80, 0.20]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.25, 0.75], dtype=float)
        return probabilities


class _DuplicateNanClassSourceModel:
    classes_ = np.array([np.nan, np.nan], dtype=object)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.5, 0.5]], dtype=float), (features.shape[0], 1))


def test_source_free_nan_class_labels_align_probability_columns() -> None:
    target_features = np.array([[-1.0, 0.0], [2.0, 0.0]], dtype=float)
    model = _NanClassSourceModel()

    result = fit_source_free_predict_proba(
        source_model=model,
        target_features=target_features,
        classes=np.array([np.nan, "stimulus"], dtype=object),
        max_iterations=0,
    )

    assert np.isnan(result.adapter.classes_[0])
    assert result.adapter.classes_[1] == "stimulus"
    assert np.allclose(result.probabilities, model.predict_proba(target_features))


def test_source_free_duplicate_nan_class_labels_are_rejected() -> None:
    target_features = np.array([[-1.0, 0.0], [2.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="classes must be unique"):
        fit_source_free_predict_proba(
            source_model=_DuplicateNanClassSourceModel(),
            target_features=target_features,
            max_iterations=0,
        )
