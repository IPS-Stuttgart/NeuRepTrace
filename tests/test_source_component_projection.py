from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_component_projection import (
    SOURCE_COMPONENT_CATEGORY,
    SourceComponentConfig,
    fit_source_component_projection,
    fit_source_component_projector,
    reconstruct_from_source_components,
    source_component_config,
    transform_with_source_components,
)


def test_source_component_projection_uses_source_rows_only() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(40, 4))
    source[:, 1] = source[:, 0] * 0.5 + rng.normal(scale=0.1, size=40)
    test = rng.normal(size=(5, 4)) + 10.0

    result = fit_source_component_projection(source_features=source, test_features=test, config={"n_components": 2})

    assert result.train_features.shape == (40, 2)
    assert result.test_features.shape == (5, 2)
    assert np.allclose(np.mean(result.train_features, axis=0), 0.0, atol=1e-6)
    assert result.metadata["source_component_protocol_category"] == SOURCE_COMPONENT_CATEGORY
    assert result.metadata["source_component_uses_test_features_for_fitting"] is False
    assert result.metadata["source_component_uses_test_labels"] is False
    assert result.metadata["source_component_valid_for_strict_source_only"] is True


def test_source_component_reconstruction_with_all_components_recovers_source() -> None:
    source = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=float)
    projector = fit_source_component_projector(source, config={"n_components": "all", "scale": False})
    scores = transform_with_source_components(source, projector)
    reconstructed = reconstruct_from_source_components(scores, projector)

    assert np.allclose(reconstructed, source, atol=1e-6)


def test_source_component_scaling_normalizes_feature_scale_before_projection() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 100.0], [2.0, 200.0], [3.0, 300.0]], dtype=float)
    test = np.asarray([[1.5, 150.0]], dtype=float)

    result = fit_source_component_projection(source_features=source, test_features=test, config={"n_components": 1, "scale": "true"})

    assert result.train_features.shape == (4, 1)
    assert result.projector.scale[1] > result.projector.scale[0]
    assert result.metadata["source_component_scale"] is True


def test_source_component_explained_variance_handles_extreme_finite_values() -> None:
    source = np.asarray(
        [
            [1e200, 0.0],
            [-1e200, 0.0],
            [0.0, 1e200],
            [0.0, -1e200],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        projector = fit_source_component_projector(source, config={"n_components": "all", "scale": False})

    assert np.all(np.isfinite(projector.explained_variance_ratio))
    assert np.allclose(projector.explained_variance_ratio, [0.5, 0.5])
    assert np.isclose(np.sum(projector.explained_variance_ratio), 1.0)


def test_source_component_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_component_projection(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_component_config_validation() -> None:
    cfg = source_component_config(n_components="2", center="false", scale="on")
    assert cfg.n_components == "2"
    assert cfg.center is False
    assert cfg.scale is True

    with pytest.raises(ValueError, match="n_components"):
        fit_source_component_projector([[0.0, 1.0], [1.0, 0.0]], config={"n_components": 0})

    with pytest.raises(ValueError, match="boolean"):
        source_component_config(center="maybe")


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        np.asarray(True),
        np.asarray([True]),
    ],
)
def test_source_component_config_rejects_boolean_component_counts(value: object) -> None:
    with pytest.raises(ValueError, match="n_components"):
        source_component_config(n_components=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        np.asarray(True),
        np.asarray([True]),
    ],
)
def test_source_component_config_rejects_boolean_epsilon(value: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        source_component_config(epsilon=value)  # type: ignore[arg-type]


def test_source_component_config_accepts_numpy_numeric_scalars() -> None:
    config = source_component_config(n_components=np.int64(1), epsilon=np.asarray(1e-6))

    assert config.n_components == 1
    assert config.epsilon == pytest.approx(1e-6)


def test_source_component_dataclass_rejects_boolean_component_count_when_used() -> None:
    config = SourceComponentConfig(n_components=True)

    with pytest.raises(ValueError, match="n_components"):
        fit_source_component_projector([[0.0, 1.0], [1.0, 0.0]], config=config)


def test_source_component_config_instance_is_normalized() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 100.0], [2.0, 200.0]], dtype=float)
    test = np.asarray([[1.5, 150.0]], dtype=float)
    config = SourceComponentConfig(n_components=1, center="false", scale="false", epsilon="1e-6")

    result = fit_source_component_projection(source_features=source, test_features=test, config=config)

    assert np.allclose(result.projector.mean, 0.0)
    assert np.allclose(result.projector.scale, 1.0)
    assert result.metadata["source_component_center"] is False
    assert result.metadata["source_component_scale"] is False
    assert result.metadata["source_component_epsilon"] == pytest.approx(1e-6)


def test_source_component_config_instance_rejects_invalid_epsilon() -> None:
    config = SourceComponentConfig(epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        fit_source_component_projector([[0.0, 1.0], [1.0, 0.0]], config=config)
