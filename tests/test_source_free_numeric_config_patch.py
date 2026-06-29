from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    ("parameter", "value", "expected_name"),
    [
        ("confidence_threshold", np.asarray(0.25), "source_free_confidence_threshold"),
        ("max_iterations", np.asarray(0), "source_free_max_iterations"),
        ("min_class_count", np.array([1]), "source_free_min_class_count"),
        ("min_active_classes", np.array([1]), "source_free_min_active_classes"),
        ("prototype_weight", np.asarray(True), "source_free_prototype_weight"),
        ("prototype_temperature", np.array([1.0]), "source_free_prototype_temperature"),
        ("balanced_topk_per_class", np.array([1]), "source_free_balanced_topk_per_class"),
    ],
)
def test_source_free_rejects_array_valued_numeric_config(parameter, value, expected_name):
    target_features = np.array([[-1.0, 0.0], [2.0, 0.0]], dtype=float)
    kwargs = {
        "source_model": _SourceModel(),
        "target_features": target_features,
        "confidence_threshold": 0.25,
        "max_iterations": 0,
        "min_class_count": 1,
        "min_active_classes": 1,
        "prototype_weight": 0.0,
        "prototype_temperature": 1.0,
        "standardize_target": False,
        "balanced_topk_per_class": None,
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=rf"{expected_name} must be"):
        fit_source_free_predict_proba(**kwargs)
