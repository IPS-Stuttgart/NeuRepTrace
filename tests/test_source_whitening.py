from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_whitening import (
    SOURCE_WHITENING_CATEGORY,
    apply_source_whitening,
    fit_source_whitening,
    fit_source_whitening_transform,
    normalize_whitening_mode,
    source_whitening_config,
)


def test_zca_whitening_is_source_only_and_centers_source_rows() -> None:
    source = np.asarray([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]], dtype=float)
    target = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=float)

    result = fit_source_whitening(source_features=source, test_features=target, config={"mode": "zca", "regularization": 1e-8})

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == target.shape
    assert np.allclose(np.mean(result.train_features, axis=0), 0.0, atol=1e-6)
    assert result.metadata["source_whitening_protocol_category"] == SOURCE_WHITENING_CATEGORY
    assert result.metadata["source_whitening_valid_for_strict_source_only"] is True
    assert result.metadata["source_whitening_uses_test_features_for_fitting"] is False
    assert result.metadata["source_whitening_uses_test_labels"] is False


def test_diagonal_whitening_matches_feature_standardization() -> None:
    source = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float)
    target = np.asarray([[6.0, 7.0]], dtype=float)

    result = fit_source_whitening(source_features=source, test_features=target, config={"mode": "diagonal", "regularization": 0.0})

    assert np.allclose(np.mean(result.train_features, axis=0), 0.0)
    assert np.allclose(np.std(result.train_features, axis=0, ddof=1), 1.0)


def test_pca_whitening_returns_feature_width_coordinates() -> None:
    source = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    target = np.asarray([[0.5, 0.5, 0.5]], dtype=float)

    result = fit_source_whitening(source_features=source, test_features=target, config={"mode": "pca"})

    assert result.train_features.shape == (4, 3)
    assert result.test_features.shape == (1, 3)
    assert result.transform.transform.shape == (3, 3)


def test_apply_source_whitening_rejects_width_mismatch() -> None:
    transform = fit_source_whitening_transform([[0.0, 0.0], [1.0, 1.0]], config={"mode": "diagonal"})

    with pytest.raises(ValueError, match="features width"):
        apply_source_whitening([[0.0, 0.0, 0.0]], transform)


def test_whitening_aliases_and_validation() -> None:
    assert normalize_whitening_mode("zscore") == "diagonal"
    assert normalize_whitening_mode("pca-whitening") == "pca"
    assert source_whitening_config(center="false").center is False

    with pytest.raises(ValueError, match="whitening mode"):
        normalize_whitening_mode("bad")

    with pytest.raises(ValueError, match="regularization"):
        source_whitening_config(regularization=-1.0)
