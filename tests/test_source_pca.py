from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_pca import (
    SOURCE_PCA_CATEGORY,
    fit_source_pca,
    fit_source_pca_projection,
    source_pca_config,
    transform_with_source_pca,
)


def test_source_pca_shapes_and_metadata() -> None:
    source = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=float,
    )
    target = np.asarray([[0.2, 0.3, 1.0], [0.8, 0.7, 1.0]], dtype=float)

    result = fit_source_pca(source_features=source, test_features=target, config={"n_components": 2})

    assert result.train_features.shape == (4, 2)
    assert result.test_features.shape == (2, 2)
    assert result.projection.components.shape == (2, 3)
    assert result.metadata["source_pca_protocol_category"] == SOURCE_PCA_CATEGORY
    assert result.metadata["source_pca_uses_test_features_for_fitting"] is False
    assert result.metadata["source_pca_uses_test_labels"] is False
    assert result.metadata["source_pca_valid_for_strict_source_only"] is True
    assert np.allclose(np.mean(result.train_features, axis=0), 0.0, atol=1e-6)


def test_source_pca_components_are_capped() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    projection = fit_source_pca_projection(source, config={"n_components": "all"})

    assert projection.components.shape == (1, 3)
    assert projection.metadata["source_pca_n_components"] == 1


def test_source_pca_scaling_and_whitening_are_finite() -> None:
    source = np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]], dtype=float)
    target = np.asarray([[1.5, 4.0]], dtype=float)

    result = fit_source_pca(source_features=source, test_features=target, config={"n_components": 1, "scale": "true", "whiten": "true"})

    assert result.train_features.shape == (4, 1)
    assert result.test_features.shape == (1, 1)
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert result.metadata["source_pca_scale"] is True
    assert result.metadata["source_pca_whiten"] is True


def test_source_pca_transform_rejects_width_mismatch() -> None:
    projection = fit_source_pca_projection([[0.0, 1.0], [1.0, 0.0]], config={"n_components": 1})

    with pytest.raises(ValueError, match="projection width"):
        transform_with_source_pca([[0.0, 1.0, 2.0]], projection)


def test_source_pca_config_validation() -> None:
    cfg = source_pca_config(n_components="all", center="yes", scale=0, whiten=1)
    assert cfg.center is True
    assert cfg.scale is False
    assert cfg.whiten is True

    with pytest.raises(ValueError, match="n_components"):
        fit_source_pca_projection([[0.0], [1.0]], config={"n_components": 0})
