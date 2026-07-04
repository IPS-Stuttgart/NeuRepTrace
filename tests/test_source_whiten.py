from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_whiten import (
    SOURCE_WHITEN_CATEGORY,
    apply_source_whiten,
    fit_source_whiten,
    fit_source_whiten_transform,
    normalize_whiten_method,
    source_whiten_config,
)


def test_pca_source_whiten_shapes_and_metadata() -> None:
    source = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [1.0, 1.0, 4.0]], dtype=float)
    test = np.asarray([[0.5, 0.5, 2.5], [2.0, 2.0, 5.0]], dtype=float)

    result = fit_source_whiten(source_features=source, test_features=test, config={"method": "pca", "n_components": 2})

    assert result.train_features.shape == (4, 2)
    assert result.test_features.shape == (2, 2)
    assert result.transform.whitening.shape == (3, 2)
    assert result.metadata["source_whiten_protocol_category"] == SOURCE_WHITEN_CATEGORY
    assert result.metadata["source_whiten_uses_test_features_for_fitting"] is False
    assert result.metadata["source_whiten_uses_test_labels"] is False
    assert result.metadata["source_whiten_valid_for_strict_source_only"] is True


def test_source_whiten_accepts_nested_one_pass_iterables() -> None:
    source = (iter(row) for row in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    test = (iter(row) for row in [[0.25, 0.75], [0.75, 0.25]])

    result = fit_source_whiten(source_features=source, test_features=test, config={"method": "pca", "n_components": 1})
    transformed = apply_source_whiten((iter(row) for row in [[0.5, 0.5]]), result.transform)

    assert result.train_features.shape == (4, 1)
    assert result.test_features.shape == (2, 1)
    assert transformed.shape == (1, 1)


def test_zca_whiten_preserves_feature_width() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    test = np.asarray([[0.25, 0.75]], dtype=float)

    result = fit_source_whiten(source_features=source, test_features=test, config={"method": "zca", "regularization": 1e-6})

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert result.transform.whitening.shape == (2, 2)


def test_apply_source_whiten_reuses_frozen_source_transform() -> None:
    source = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=float)
    transform = fit_source_whiten_transform(source, config={"method": "pca", "n_components": "all"})

    manual = apply_source_whiten([[1.0, 1.0]], transform)

    assert manual.shape == (1, 2)
    assert np.allclose(manual, np.zeros((1, 2)), atol=1e-6)


def test_source_whiten_aliases_and_validation() -> None:
    assert normalize_whiten_method("pca-whiten") == "pca"
    assert normalize_whiten_method("zca_whiten") == "zca"
    assert source_whiten_config(center="false").center is False

    with pytest.raises(ValueError, match="whitening method"):
        normalize_whiten_method("bad")

    with pytest.raises(ValueError, match="ZCA"):
        fit_source_whiten_transform([[0.0, 0.0], [1.0, 1.0]], config={"method": "zca", "n_components": 1})


def test_source_whiten_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_whiten(source_features=[[0.0, 1.0]], test_features=[[0.0]])
