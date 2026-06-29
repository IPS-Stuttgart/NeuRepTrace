from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_zca import (
    SOURCE_ZCA_CATEGORY,
    SourceZCAConfig,
    apply_source_zca_transform,
    fit_source_zca_reference,
    fit_source_zca_transform,
    source_zca_config,
)


def test_source_zca_shapes_and_metadata() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    test = np.asarray([[0.5, 0.5], [2.0, 0.0]], dtype=float)

    result = fit_source_zca_transform(source_features=source, test_features=test)

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert result.reference.whitening.shape == (2, 2)
    assert result.metadata["source_zca_protocol_category"] == SOURCE_ZCA_CATEGORY
    assert result.metadata["source_zca_uses_source_features"] is True
    assert result.metadata["source_zca_uses_test_features_for_fitting"] is False
    assert result.metadata["source_zca_uses_test_labels"] is False
    assert result.metadata["source_zca_valid_for_strict_source_only"] is True


def test_source_zca_reference_can_be_reused() -> None:
    source = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float)
    test = np.asarray([[1.0, 2.0]], dtype=float)
    reference = fit_source_zca_reference(source)

    direct = apply_source_zca_transform(test, reference)
    via_fit = fit_source_zca_transform(source_features=source, test_features=test)

    assert np.allclose(direct, via_fit.test_features)


def test_source_zca_recolor_approximately_restores_centered_source() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.2], [0.2, 1.0], [1.0, 1.0]], dtype=float)
    result = fit_source_zca_transform(source_features=source, test_features=source, config={"recolor": True})

    assert np.allclose(result.train_features, source - source.mean(axis=0), atol=1e-4)
    assert result.metadata["source_zca_recolor"] is True


def test_source_zca_config_validation() -> None:
    cfg = source_zca_config(regularization="1e-4", center="true", recolor="false")
    assert np.isclose(cfg.regularization, 1e-4)
    assert cfg.center is True
    assert cfg.recolor is False

    with pytest.raises(ValueError, match="regularization"):
        source_zca_config(regularization=0.0)


def test_source_zca_revalidates_direct_config_objects() -> None:
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    result = fit_source_zca_transform(
        source_features=source,
        test_features=source,
        config=SourceZCAConfig(regularization="1e-4", center="false", recolor="1"),  # type: ignore[arg-type]
    )

    assert np.isclose(result.reference.config.regularization, 1e-4)
    assert result.reference.config.center is False
    assert result.reference.config.recolor is True
    assert np.allclose(result.reference.mean, np.zeros(source.shape[1]))
    assert result.metadata["source_zca_center"] is False
    assert result.metadata["source_zca_recolor"] is True


def test_source_zca_rejects_invalid_direct_config_objects() -> None:
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    with pytest.raises(ValueError, match="regularization"):
        fit_source_zca_reference(source, config=SourceZCAConfig(regularization=0.0))

    with pytest.raises(ValueError, match="center"):
        fit_source_zca_reference(source, config=SourceZCAConfig(center="maybe"))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="recolor"):
        fit_source_zca_reference(source, config=SourceZCAConfig(recolor=np.asarray([True])))  # type: ignore[arg-type]


def test_source_zca_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_zca_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_zca_transform(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )
