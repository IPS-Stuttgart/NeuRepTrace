from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_rff import (
    SOURCE_RFF_CATEGORY,
    SourceRFFConfig,
    apply_source_rff,
    fit_source_rff_reference,
    fit_source_rff_transform,
    normalize_gamma,
    source_rff_config,
)


def test_source_rff_transform_shapes_and_metadata() -> None:
    source = np.arange(20, dtype=float).reshape(5, 4)
    test = np.arange(8, dtype=float).reshape(2, 4)

    result = fit_source_rff_transform(
        source_features=source,
        test_features=test,
        config={"n_components": 16, "gamma": "scale", "random_state": 7},
    )

    assert result.train_features.shape == (5, 16)
    assert result.test_features.shape == (2, 16)
    assert result.reference.weights.shape == (4, 16)
    assert result.metadata["source_rff_protocol_category"] == SOURCE_RFF_CATEGORY
    assert result.metadata["source_rff_uses_source_features"] is True
    assert result.metadata["source_rff_uses_test_features_for_fitting"] is False
    assert result.metadata["source_rff_uses_test_labels"] is False
    assert result.metadata["source_rff_valid_for_strict_source_only"] is True


def test_source_rff_is_reproducible_with_fixed_seed() -> None:
    source = np.arange(12, dtype=float).reshape(3, 4)
    first = fit_source_rff_reference(source, config={"n_components": 8, "random_state": 42})
    second = fit_source_rff_reference(source, config={"n_components": 8, "random_state": 42})

    assert np.allclose(first.weights, second.weights)
    assert np.allclose(first.phase, second.phase)


def test_source_rff_reference_can_be_reused() -> None:
    source = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float)
    test = np.asarray([[1.0, 2.0]], dtype=float)
    reference = fit_source_rff_reference(source, config={"n_components": 4, "gamma": 0.5, "random_state": 3})

    direct = apply_source_rff(test, reference)
    via_fit = fit_source_rff_transform(source_features=source, test_features=test, config={"n_components": 4, "gamma": 0.5, "random_state": 3})

    assert np.allclose(direct, via_fit.test_features)


def test_source_rff_standardization_records_source_statistics() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 12.0], [4.0, 14.0]], dtype=float)
    reference = fit_source_rff_reference(source, config={"n_components": 4, "standardize": True})

    assert np.allclose(reference.mean, np.asarray([2.0, 12.0]))
    assert np.all(reference.scale > 0.0)
    assert reference.config.standardize is True


def test_gamma_aliases_and_config_validation() -> None:
    assert normalize_gamma("auto") == "auto"
    assert normalize_gamma("scale") == "scale"
    assert normalize_gamma("0.25") == 0.25
    cfg = source_rff_config(n_components="8", standardize="true", epsilon="1e-6")
    assert cfg.n_components == 8
    assert cfg.standardize is True
    assert np.isclose(cfg.epsilon, 1e-6)

    with pytest.raises(ValueError, match="gamma"):
        normalize_gamma(0.0)

    with pytest.raises(ValueError, match="n_components"):
        fit_source_rff_reference([[0.0], [1.0]], config={"n_components": 0})


def test_source_rff_accepts_scalar_numpy_array_config_values() -> None:
    cfg = source_rff_config(
        n_components=np.asarray(8),  # type: ignore[arg-type]
        gamma=np.asarray(0.25),  # type: ignore[arg-type]
        random_state=np.asarray(3),  # type: ignore[arg-type]
        standardize=np.asarray(True),  # type: ignore[arg-type]
        epsilon=np.asarray(1e-6),  # type: ignore[arg-type]
    )

    assert cfg.n_components == 8
    assert cfg.gamma == 0.25
    assert cfg.random_state == 3
    assert cfg.standardize is True
    assert np.isclose(cfg.epsilon, 1e-6)

    source = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]], dtype=float)
    direct = SourceRFFConfig(
        n_components=np.asarray(4),  # type: ignore[arg-type]
        gamma=np.asarray(0.5),  # type: ignore[arg-type]
        random_state=np.asarray(7),  # type: ignore[arg-type]
        standardize=np.asarray(False),  # type: ignore[arg-type]
        epsilon=np.asarray(1e-6),  # type: ignore[arg-type]
    )
    reference = fit_source_rff_reference(source, config=direct)

    assert reference.config.n_components == 4
    assert reference.config.gamma == 0.5
    assert reference.config.random_state == 7
    assert reference.config.standardize is False
    assert np.isclose(reference.config.epsilon, 1e-6)


def test_source_rff_revalidates_direct_config_instances() -> None:
    source = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]], dtype=float)
    cfg = SourceRFFConfig(
        n_components="4",
        gamma="0.5",  # type: ignore[arg-type]
        random_state="3",  # type: ignore[arg-type]
        standardize="false",  # type: ignore[arg-type]
        epsilon="1e-6",  # type: ignore[arg-type]
    )

    reference = fit_source_rff_reference(source, config=cfg)

    assert reference.config.n_components == 4
    assert reference.config.gamma == 0.5
    assert reference.config.random_state == 3
    assert reference.config.standardize is False
    assert np.isclose(reference.config.epsilon, 1e-6)
    assert np.allclose(reference.mean, np.zeros(source.shape[1]))
    assert np.allclose(reference.scale, np.ones(source.shape[1]))


def test_source_rff_rejects_bool_and_array_numeric_controls() -> None:
    invalid_configs = [
        {"n_components": True},
        {"n_components": np.asarray([4])},
        {"gamma": True},
        {"gamma": np.asarray([0.5])},
        {"random_state": True},
        {"random_state": np.asarray([7])},
        {"standardize": np.asarray([True])},
        {"epsilon": True},
        {"epsilon": np.asarray([1e-6])},
    ]

    for config in invalid_configs:
        with pytest.raises(ValueError):
            source_rff_config(**config)  # type: ignore[arg-type]


def test_source_rff_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_rff_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_rff_transform(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )