from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.geodesic_flow import (
    GEODESIC_FLOW_CATEGORY,
    fit_sampled_geodesic_flow_features,
    sample_geodesic_bases,
    transform_with_geodesic_bases,
)


def test_sampled_geodesic_flow_shapes_and_metadata() -> None:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    target = source @ np.asarray([[0.8, 0.2, 0.0], [-0.1, 0.9, 0.1], [0.2, 0.0, 1.0]]) + 1.5

    result = fit_sampled_geodesic_flow_features(
        source_features=source,
        target_test_features=target,
        config={"n_components": 2, "n_steps": 4},
    )

    assert result.train_features.shape == (5, 8)
    assert result.test_features.shape == (5, 8)
    assert len(result.bases) == 4
    assert result.metadata["geodesic_flow_protocol_category"] == GEODESIC_FLOW_CATEGORY
    assert result.metadata["geodesic_flow_uses_target_features"] is True
    assert result.metadata["geodesic_flow_uses_target_labels"] is False
    assert result.metadata["geodesic_flow_transductive"] is True
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))


def test_sampled_geodesic_flow_uses_separate_target_adaptation_rows() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    target_adaptation = source + 2.0
    target_test = np.asarray([[2.0, 2.0], [3.0, 2.0]], dtype=float)

    result = fit_sampled_geodesic_flow_features(
        source_features=source,
        target_adaptation_features=target_adaptation,
        target_test_features=target_test,
        config={"n_components": "all", "n_steps": 3},
    )

    assert result.train_features.shape[1] == result.test_features.shape[1]
    assert result.metadata["geodesic_flow_transductive"] is False
    assert result.metadata["geodesic_flow_target_feature_source"] == "target_adaptation_features"


def test_sampled_bases_are_orthonormal_and_include_endpoints() -> None:
    source_components = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target_components = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    bases = sample_geodesic_bases(source_components, target_components, n_steps=5, include_endpoints=True)

    assert [basis.position for basis in bases] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    for basis in bases:
        gram = basis.basis @ basis.basis.T
        assert np.allclose(gram, np.eye(2), atol=1e-6)


def test_geodesic_transform_normalizes_blocks() -> None:
    bases = sample_geodesic_bases([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], n_steps=2)
    features = np.asarray([[1.0, 2.0]], dtype=float)

    normalized = transform_with_geodesic_bases(features, bases, normalize_blocks=True)
    raw = transform_with_geodesic_bases(features, bases, normalize_blocks=False)

    assert np.allclose(raw, np.asarray([[1.0, 2.0, 1.0, 2.0]]))
    assert np.allclose(normalized, raw / np.sqrt(2.0))


def test_geodesic_components_are_capped() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result = fit_sampled_geodesic_flow_features(
        source_features=source,
        target_test_features=target,
        config={"n_components": "all", "n_steps": 2},
    )

    assert result.metadata["geodesic_flow_n_components"] == 1
    assert result.train_features.shape == (2, 2)


def test_geodesic_flow_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_sampled_geodesic_flow_features(
            source_features=[[0.0, 1.0]],
            target_test_features=[[0.0, 1.0, 2.0]],
        )
