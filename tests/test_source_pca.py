from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_pca import (
    SOURCE_PCA_CATEGORY,
    apply_source_pca,
    fit_source_pca,
    fit_source_pca_projection,
    source_pca_config,
)


def test_source_pca_projects_source_and_test_rows_without_target_fit() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
    test = np.asarray([[0.5, 0.5, 1.0], [2.0, 2.0, 0.0]], dtype=float)

    result = fit_source_pca(source_features=source, test_features=test, config={"n_components": 2})

    assert result.train_features.shape == (4, 2)
    assert result.test_features.shape == (2, 2)
    assert result.projection.components.shape == (2, 3)
    assert result.metadata["source_pca_protocol_category"] == SOURCE_PCA_CATEGORY
    assert result.metadata["source_pca_uses_test_features_for_fitting"] is False
    assert result.metadata["source_pca_uses_test_labels"] is False
    assert result.metadata["source_pca_valid_for_strict_source_only"] is True


def test_source_pca_components_are_capped_by_rank() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    test = np.asarray([[0.5, 0.0, 0.0]], dtype=float)

    result = fit_source_pca(source_features=source, test_features=test, config={"n_components": "all"})

    assert result.train_features.shape == (2, 1)
    assert result.metadata["source_pca_n_components"] == 1


def test_apply_source_pca_reuses_projection() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    projection = fit_source_pca_projection(source, config={"n_components": 1})

    first = apply_source_pca([[0.25, 0.25]], projection)
    second = apply_source_pca([[0.25, 0.25]], projection)

    assert first.shape == (1, 1)
    assert np.allclose(first, second)


def test_source_pca_scaling_and_whitening_are_finite() -> None:
    source = np.asarray([[0.0, 10.0], [1.0, 12.0], [2.0, 14.0], [3.0, 16.0]], dtype=float)
    test = np.asarray([[1.5, 13.0]], dtype=float)

    result = fit_source_pca(source_features=source, test_features=test, config={"n_components": 1, "scale": True, "whiten": True})

    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert result.metadata["source_pca_scale"] is True
    assert result.metadata["source_pca_whiten"] is True


def test_source_pca_config_validation() -> None:
    assert source_pca_config(center="false").center is False
    assert source_pca_config(scale="yes").scale is True

    with pytest.raises(ValueError, match="n_components"):
        fit_source_pca_projection([[0.0, 1.0]], config={"n_components": 0})

    with pytest.raises(ValueError, match="boolean"):
        source_pca_config(whiten="maybe")


def test_source_pca_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_pca(source_features=[[0.0, 1.0]], test_features=[[0.0]], config={"n_components": 1})
