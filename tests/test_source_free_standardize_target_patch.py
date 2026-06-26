from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free import SourceFreeSubjectAdapter


class _ToySourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        x = np.asarray(features, dtype=float)
        probabilities = np.tile(np.array([[0.8, 0.2]], dtype=float), (x.shape[0], 1))
        probabilities[x[:, 0] > np.mean(x[:, 0])] = np.array([0.2, 0.8], dtype=float)
        return probabilities


def _target_features() -> np.ndarray:
    return np.array(
        [
            [10.0, 2.0],
            [12.0, 4.0],
            [14.0, 6.0],
            [16.0, 8.0],
        ],
        dtype=float,
    )


def _fit_adapter(standardize_target):
    return SourceFreeSubjectAdapter(
        source_model=_ToySourceModel(),
        standardize_target=standardize_target,
        max_iterations=0,
        feature_space="input",
    ).fit(_target_features())


@pytest.mark.parametrize("value", ["false", "False", "0", "off", "no", 0, False, np.bool_(False)])
def test_source_free_standardize_target_false_aliases_disable_standardization(value):
    adapter = _fit_adapter(value)

    assert adapter.metadata()["source_free_standardize_target"] is False
    assert np.allclose(adapter.target_embedding_mean_, np.zeros((1, 2)))
    assert np.allclose(adapter.target_embedding_scale_, np.ones((1, 2)))


@pytest.mark.parametrize("value", ["true", "True", "1", "on", "yes", 1, True, np.bool_(True)])
def test_source_free_standardize_target_true_aliases_enable_standardization(value):
    target = _target_features()
    adapter = _fit_adapter(value)

    assert adapter.metadata()["source_free_standardize_target"] is True
    assert np.allclose(adapter.target_embedding_mean_, target.mean(axis=0, keepdims=True))
    assert np.allclose(adapter.target_embedding_scale_, target.std(axis=0, keepdims=True))


@pytest.mark.parametrize("value", ["", "maybe", 2, -1])
def test_source_free_standardize_target_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match="source_free_standardize_target must be a boolean value"):
        _fit_adapter(value)
